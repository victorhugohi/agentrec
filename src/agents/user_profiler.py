"""User profiler agent that builds and maintains user preference profiles."""

from __future__ import annotations

import logging
from typing import Any

import asyncpg
import pandas as pd

from src.agents.base import BaseAgent
from src.models.embeddings import EmbeddingStore, get_embedding_store
from src.models.schemas import Rating, UserProfileInput, UserProfileOutput

logger = logging.getLogger(__name__)

# Rating threshold — only items rated at or above this contribute to the user embedding.
_HIGH_RATING_THRESHOLD: float = 0.7  # on the normalised [0, 1] scale

# Mock database used when no real data is loaded.
_MOCK_RATINGS: dict[int, list[dict[str, Any]]] = {
    1: [
        {"item_id": 1, "rating": 5.0, "title": "Toy Story", "genres": ["Animation", "Comedy"]},
        {"item_id": 50, "rating": 4.5, "title": "Star Wars", "genres": ["Action", "Sci-Fi"]},
        {"item_id": 110, "rating": 4.0, "title": "Braveheart", "genres": ["Drama", "War"]},
        {"item_id": 260, "rating": 2.0, "title": "The Thing", "genres": ["Horror", "Sci-Fi"]},
        {"item_id": 296, "rating": 4.5, "title": "Pulp Fiction", "genres": ["Crime", "Drama"]},
    ],
    2: [
        {"item_id": 1, "rating": 3.0, "title": "Toy Story", "genres": ["Animation", "Comedy"]},
        {"item_id": 32, "rating": 4.0, "title": "Twelve Monkeys", "genres": ["Sci-Fi", "Thriller"]},
        {"item_id": 47, "rating": 5.0, "title": "Seven", "genres": ["Crime", "Thriller"]},
        {"item_id": 150, "rating": 3.5, "title": "Apollo 13", "genres": ["Drama", "Adventure"]},
    ],
    3: [
        {"item_id": 1, "rating": 4.0, "title": "Toy Story", "genres": ["Animation", "Comedy"]},
        {"item_id": 110, "rating": 5.0, "title": "Braveheart", "genres": ["Drama", "War"]},
        {"item_id": 150, "rating": 4.5, "title": "Apollo 13", "genres": ["Drama", "Adventure"]},
    ],
}


class UserProfilerAgent(BaseAgent):
    """Builds user preference profiles from interaction history.

    Supports three data modes (checked in order):

    1. **Database** — when *db_pool* is provided, ratings are queried from
       PostgreSQL via ``fetch_user_ratings``.
    2. **DataFrame** — when *ratings_df* / *movies_df* are provided (e.g.
       from the data loader), profiles use in-memory DataFrames.
    3. **Mock** — falls back to built-in mock data for development.

    Attributes:
        _embedding_store: Shared embedding store for looking up item vectors.
        _db_pool: asyncpg connection pool (or None).
        _ratings_df: Real ratings DataFrame (or None).
        _movies_df: Real movies DataFrame (or None).
    """

    def __init__(
        self,
        embedding_store: EmbeddingStore | None = None,
        db_pool: asyncpg.Pool | None = None,
        ratings_df: pd.DataFrame | None = None,
        movies_df: pd.DataFrame | None = None,
    ) -> None:
        """Initialize the user profiler agent.

        Args:
            embedding_store: Optional shared embedding store.
            db_pool: asyncpg pool for production DB queries.
            ratings_df: Processed ratings DataFrame (fallback).
            movies_df: Processed movies DataFrame (fallback).
        """
        super().__init__(name="user_profiler")
        self._embedding_store = embedding_store or get_embedding_store()
        self._db_pool = db_pool
        self._ratings_df = ratings_df
        self._movies_df = movies_df

        if self._db_pool is not None:
            logger.info("UserProfiler: using database")
        elif self._ratings_df is not None:
            logger.info(
                "UserProfiler: using DataFrame (%d ratings, %d movies)",
                len(self._ratings_df),
                len(self._movies_df) if self._movies_df is not None else 0,
            )
        else:
            logger.info("UserProfiler: using mock data")

    @property
    def has_real_data(self) -> bool:
        """Return whether the agent is backed by real data (DB or DataFrame)."""
        return self._db_pool is not None or self._ratings_df is not None

    async def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build a preference profile for the given user.

        Args:
            payload: Must conform to :class:`UserProfileInput`
                (contains ``user_id``).

        Returns:
            A dict matching :class:`UserProfileOutput` with ``user_id``,
            ``ratings``, ``embedding``, ``rating_count``, and ``avg_rating``.
        """
        input_data = UserProfileInput(**payload)
        logger.info("Building profile for user %d", input_data.user_id)

        if self._db_pool is not None:
            ratings = await self._load_db_ratings(input_data.user_id)
        elif self._ratings_df is not None:
            ratings = self._load_real_ratings(input_data.user_id)
        else:
            ratings = self._load_mock_ratings(input_data.user_id)

        logger.debug("Found %d ratings for user %d", len(ratings), input_data.user_id)

        embedding = self._compute_embedding(ratings)

        avg_rating = sum(r.rating for r in ratings) / len(ratings) if ratings else 0.0

        output = UserProfileOutput(
            user_id=input_data.user_id,
            ratings=ratings,
            embedding=embedding,
            rating_count=len(ratings),
            avg_rating=round(avg_rating, 4),
        )
        logger.info(
            "User %d profile: %d ratings, avg=%.4f, embedding_dim=%d",
            output.user_id,
            output.rating_count,
            output.avg_rating,
            len(output.embedding),
        )
        return output.model_dump()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def _load_db_ratings(self, user_id: int) -> list[Rating]:
        """Query ratings for *user_id* from PostgreSQL."""
        from src.data.database import fetch_user_ratings

        rows = await fetch_user_ratings(self._db_pool, user_id)
        return [
            Rating(
                item_id=row["movie_id"],
                rating=row["rating"],
                title=row["title"],
                genres=row["genres"],
            )
            for row in rows
        ]

    def _load_real_ratings(self, user_id: int) -> list[Rating]:
        """Query ratings for *user_id* from the real DataFrame."""
        assert self._ratings_df is not None
        user_rows = self._ratings_df[self._ratings_df["userId"] == user_id]
        if user_rows.empty:
            return []

        # Resolve movie titles if movies_df is available
        if self._movies_df is not None:
            merged = user_rows.merge(self._movies_df, on="movieId", how="left")
        else:
            merged = user_rows.copy()
            merged["title"] = ""
            merged["genres"] = ""

        ratings: list[Rating] = []
        for _, row in merged.iterrows():
            genres_raw = row.get("genres", "")
            if isinstance(genres_raw, str):
                genres = [g for g in genres_raw.split("|") if g and g != "(no genres listed)"]
            else:
                genres = genres_raw if isinstance(genres_raw, list) else []

            ratings.append(
                Rating(
                    item_id=int(row["movieId"]),
                    rating=float(row["rating"]),
                    title=str(row.get("title", "")),
                    genres=genres,
                )
            )
        return ratings

    @staticmethod
    def _load_mock_ratings(user_id: int) -> list[Rating]:
        """Return mock ratings for *user_id*."""
        raw = _MOCK_RATINGS.get(user_id, [])
        return [Rating(**r) for r in raw]

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    def _compute_embedding(self, ratings: list[Rating]) -> list[float]:
        """Compute a user embedding from highly-rated item embeddings.

        Looks up each item's vector via the shared :class:`EmbeddingStore`,
        then takes the element-wise mean across items the user rated at or
        above ``_HIGH_RATING_THRESHOLD``.

        Args:
            ratings: The user's rating history.

        Returns:
            A list of floats representing the user embedding.  Returns a
            zero vector of length ``embedding_dim`` if no qualifying ratings
            exist.
        """
        dim = self._embedding_store.embedding_dim
        threshold = _HIGH_RATING_THRESHOLD if self.has_real_data else 3.5
        high_rated = [r for r in ratings if r.rating >= threshold]

        if not high_rated:
            return [0.0] * dim

        item_ids = [r.item_id for r in high_rated]
        vectors = self._embedding_store.item_embeddings_batch(item_ids)

        mean_embedding = [round(sum(col) / len(vectors), 4) for col in zip(*vectors, strict=True)]
        return mean_embedding
