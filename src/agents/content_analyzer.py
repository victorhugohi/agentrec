"""Content analyzer agent that extracts and manages item features."""

from __future__ import annotations

import logging
import re
from typing import Any

import asyncpg
import pandas as pd

from src.agents.base import BaseAgent
from src.models.embeddings import EmbeddingStore, get_embedding_store
from src.models.schemas import ContentAnalyzerInput, ContentAnalyzerOutput, ContentItem

logger = logging.getLogger(__name__)

# Mock movie catalog used when no real data is loaded.
_MOCK_CATALOG: list[dict[str, Any]] = [
    {"item_id": 1, "title": "Toy Story", "genres": ["Animation", "Comedy"], "year": 1995, "tags": ["pixar", "fun"]},
    {"item_id": 32, "title": "Twelve Monkeys", "genres": ["Sci-Fi", "Thriller"], "year": 1995, "tags": ["time-travel", "dystopia"]},
    {"item_id": 47, "title": "Seven", "genres": ["Crime", "Thriller"], "year": 1995, "tags": ["dark", "mystery"]},
    {"item_id": 50, "title": "Star Wars", "genres": ["Action", "Sci-Fi"], "year": 1977, "tags": ["space", "classic"]},
    {"item_id": 110, "title": "Braveheart", "genres": ["Drama", "War"], "year": 1995, "tags": ["epic", "historical"]},
    {"item_id": 150, "title": "Apollo 13", "genres": ["Drama", "Adventure"], "year": 1995, "tags": ["space", "based-on-true-story"]},
    {"item_id": 260, "title": "The Thing", "genres": ["Horror", "Sci-Fi"], "year": 1982, "tags": ["alien", "suspense"]},
    {"item_id": 296, "title": "Pulp Fiction", "genres": ["Crime", "Drama"], "year": 1994, "tags": ["tarantino", "nonlinear"]},
    {"item_id": 318, "title": "Shawshank Redemption", "genres": ["Drama"], "year": 1994, "tags": ["prison", "hope"]},
    {"item_id": 356, "title": "Forrest Gump", "genres": ["Comedy", "Drama"], "year": 1994, "tags": ["heartwarming", "classic"]},
]
_CATALOG_INDEX: dict[int, dict[str, Any]] = {m["item_id"]: m for m in _MOCK_CATALOG}

# Regex to extract year from movie title, e.g. "Toy Story (1995)"
_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


class ContentAnalyzerAgent(BaseAgent):
    """Analyzes item features and metadata for candidate generation.

    Supports three data modes (checked in order):

    1. **Database** — when *db_pool* is provided, movies are queried from
       PostgreSQL.
    2. **DataFrame** — when *movies_df* is provided (e.g. from the data
       loader), item metadata comes from the in-memory DataFrame.
    3. **Mock** — falls back to the built-in mock catalog.

    Attributes:
        _embedding_store: Shared embedding store for looking up item vectors.
        _db_pool: asyncpg connection pool (or None).
        _movies_df: Real movies DataFrame (or None for mock mode).
        _movie_index: Dict mapping movieId → row dict for fast DataFrame lookup.
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore | None = None,
        db_pool: asyncpg.Pool | None = None,
        movies_df: pd.DataFrame | None = None,
    ) -> None:
        """Initialize the content analyzer agent.

        Args:
            embedding_store: Optional shared embedding store.
            db_pool: asyncpg pool for production DB queries.
            movies_df: Processed movies DataFrame (fallback).
        """
        super().__init__(name="content_analyzer")
        self._embedding_store = embedding_store or get_embedding_store()
        self._db_pool = db_pool
        self._movies_df = movies_df
        self._movie_index: dict[int, dict[str, Any]] = {}

        if self._db_pool is not None:
            logger.info("ContentAnalyzer: using database")
        elif self._movies_df is not None:
            self._movie_index = self._build_index(self._movies_df)
            logger.info(
                "ContentAnalyzer: using DataFrame (%d movies)",
                len(self._movie_index),
            )
        else:
            logger.info("ContentAnalyzer: using mock data")

    @property
    def has_real_data(self) -> bool:
        """Return whether the agent is backed by real data (DB or DataFrame)."""
        return self._db_pool is not None or self._movies_df is not None

    async def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Analyze content features for candidate items.

        Args:
            payload: Must conform to :class:`ContentAnalyzerInput`.
                Contains ``user_id`` and optionally ``item_ids`` to restrict
                the candidate set.

        Returns:
            A dict matching :class:`ContentAnalyzerOutput` with ``user_id``
            and ``items`` (list of :class:`ContentItem` dicts with embeddings).
        """
        input_data = ContentAnalyzerInput(**payload)
        logger.info(
            "Analyzing content for user %d (requested %d specific items)",
            input_data.user_id, len(input_data.item_ids),
        )

        if self._db_pool is not None:
            items = await self._load_db_items(input_data.item_ids)
        elif self.has_real_data:
            items = self._load_real_items(input_data.item_ids)
        else:
            items = self._load_mock_items(input_data.item_ids)

        logger.info(
            "Returning %d candidate items (embedding_dim=%d) for user %d",
            len(items), self._embedding_store.embedding_dim, input_data.user_id,
        )

        output = ContentAnalyzerOutput(user_id=input_data.user_id, items=items)
        return output.model_dump()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def _load_db_items(self, item_ids: list[int]) -> list[ContentItem]:
        """Load items from PostgreSQL."""
        from src.data.database import fetch_all_movies, fetch_movies_by_ids

        if item_ids:
            rows = await fetch_movies_by_ids(self._db_pool, item_ids)
        else:
            rows = await fetch_all_movies(self._db_pool)

        items: list[ContentItem] = []
        for row in rows:
            embedding = self._embedding_store.item_embedding(row["movie_id"])
            items.append(ContentItem(
                item_id=row["movie_id"],
                title=row["title"],
                genres=row["genres"],
                year=row["year"],
                tags=[],
                embedding=embedding,
            ))
        return items

    def _load_real_items(self, item_ids: list[int]) -> list[ContentItem]:
        """Load items from the real movie index."""
        if item_ids:
            raw_items = [
                self._movie_index[iid]
                for iid in item_ids
                if iid in self._movie_index
            ]
        else:
            raw_items = list(self._movie_index.values())

        items: list[ContentItem] = []
        for raw in raw_items:
            embedding = self._embedding_store.item_embedding(raw["item_id"])
            items.append(ContentItem(embedding=embedding, **raw))
        return items

    def _load_mock_items(self, item_ids: list[int]) -> list[ContentItem]:
        """Load items from the mock catalog."""
        if item_ids:
            raw_items = [
                _CATALOG_INDEX[iid] for iid in item_ids if iid in _CATALOG_INDEX
            ]
        else:
            raw_items = _MOCK_CATALOG

        items: list[ContentItem] = []
        for raw in raw_items:
            embedding = self._embedding_store.item_embedding(raw["item_id"])
            items.append(ContentItem(embedding=embedding, **raw))
        return items

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_index(movies_df: pd.DataFrame) -> dict[int, dict[str, Any]]:
        """Build a dict mapping movieId → metadata dict from the DataFrame."""
        index: dict[int, dict[str, Any]] = {}
        for _, row in movies_df.iterrows():
            movie_id = int(row["movieId"])
            title = str(row.get("title", ""))

            genres_raw = row.get("genres", "")
            if isinstance(genres_raw, str):
                genres = [g for g in genres_raw.split("|") if g and g != "(no genres listed)"]
            else:
                genres = genres_raw if isinstance(genres_raw, list) else []

            # Extract year from title
            year_match = _YEAR_RE.search(title)
            year = int(year_match.group(1)) if year_match else 0

            index[movie_id] = {
                "item_id": movie_id,
                "title": title,
                "genres": genres,
                "year": year,
                "tags": [],
            }
        return index
