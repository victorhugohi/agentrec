"""Tests for agents backed by real (synthetic) MovieLens data."""

import pandas as pd
import pytest

from src.agents.content_analyzer import ContentAnalyzerAgent
from src.agents.user_profiler import UserProfilerAgent
from src.models.embeddings import EmbeddingStore

# ── Fixtures ─────────────────────────────────────────────────────────

EMBEDDING_DIM = 32


@pytest.fixture
def store() -> EmbeddingStore:
    return EmbeddingStore(embedding_dim=EMBEDDING_DIM)


@pytest.fixture
def ratings_df() -> pd.DataFrame:
    """Synthetic ratings with contiguous 0-based IDs and normalised ratings."""
    return pd.DataFrame(
        {
            "userId": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "movieId": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            "rating": [1.0, 0.8, 0.6, 0.4, 0.2, 0.5, 0.9, 0.3, 0.7, 0.1],
            "timestamp": [100, 200, 300, 400, 500, 110, 210, 310, 410, 510],
        }
    )


@pytest.fixture
def movies_df() -> pd.DataFrame:
    """Synthetic movies with contiguous 0-based IDs."""
    return pd.DataFrame(
        {
            "movieId": [0, 1, 2, 3, 4],
            "title": [
                "Alpha (2000)",
                "Beta (2001)",
                "Gamma (2002)",
                "Delta (2003)",
                "Epsilon (2004)",
            ],
            "genres": [
                "Action|Comedy",
                "Drama|Thriller",
                "Sci-Fi",
                "Romance|Drama",
                "Horror",
            ],
        }
    )


@pytest.fixture
def real_profiler(
    store: EmbeddingStore,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> UserProfilerAgent:
    return UserProfilerAgent(
        embedding_store=store,
        ratings_df=ratings_df,
        movies_df=movies_df,
    )


@pytest.fixture
def real_analyzer(
    store: EmbeddingStore,
    movies_df: pd.DataFrame,
) -> ContentAnalyzerAgent:
    return ContentAnalyzerAgent(
        embedding_store=store,
        movies_df=movies_df,
    )


# ── UserProfiler with real data ──────────────────────────────────────


class TestUserProfilerRealData:
    @pytest.mark.asyncio
    async def test_has_real_data(self, real_profiler: UserProfilerAgent) -> None:
        assert real_profiler.has_real_data is True

    @pytest.mark.asyncio
    async def test_known_user_returns_ratings(self, real_profiler: UserProfilerAgent) -> None:
        result = await real_profiler.run({"user_id": 0})
        assert result["user_id"] == 0
        assert result["rating_count"] == 5
        assert len(result["ratings"]) == 5

    @pytest.mark.asyncio
    async def test_unknown_user_returns_empty(self, real_profiler: UserProfilerAgent) -> None:
        result = await real_profiler.run({"user_id": 999})
        assert result["rating_count"] == 0
        assert result["embedding"] == [0.0] * EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_embedding_dim_matches_store(self, real_profiler: UserProfilerAgent) -> None:
        result = await real_profiler.run({"user_id": 0})
        assert len(result["embedding"]) == EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_ratings_include_title_and_genres(self, real_profiler: UserProfilerAgent) -> None:
        result = await real_profiler.run({"user_id": 0})
        first = result["ratings"][0]
        assert first["title"] != ""
        assert isinstance(first["genres"], list)

    @pytest.mark.asyncio
    async def test_high_rated_items_contribute_to_embedding(
        self, real_profiler: UserProfilerAgent
    ) -> None:
        """User 0 has items rated 1.0 and 0.8 (above 0.7 threshold) → non-zero embedding."""
        result = await real_profiler.run({"user_id": 0})
        assert any(x != 0.0 for x in result["embedding"])

    @pytest.mark.asyncio
    async def test_different_users_different_profiles(
        self, real_profiler: UserProfilerAgent
    ) -> None:
        r0 = await real_profiler.run({"user_id": 0})
        r1 = await real_profiler.run({"user_id": 1})
        assert r0["avg_rating"] != r1["avg_rating"]


# ── ContentAnalyzer with real data ───────────────────────────────────


class TestContentAnalyzerRealData:
    @pytest.mark.asyncio
    async def test_has_real_data(self, real_analyzer: ContentAnalyzerAgent) -> None:
        assert real_analyzer.has_real_data is True

    @pytest.mark.asyncio
    async def test_full_catalog(self, real_analyzer: ContentAnalyzerAgent) -> None:
        result = await real_analyzer.run({"user_id": 0})
        assert len(result["items"]) == 5

    @pytest.mark.asyncio
    async def test_filter_by_ids(self, real_analyzer: ContentAnalyzerAgent) -> None:
        result = await real_analyzer.run({"user_id": 0, "item_ids": [0, 2]})
        ids = {item["item_id"] for item in result["items"]}
        assert ids == {0, 2}

    @pytest.mark.asyncio
    async def test_missing_id_skipped(self, real_analyzer: ContentAnalyzerAgent) -> None:
        result = await real_analyzer.run({"user_id": 0, "item_ids": [0, 999]})
        assert len(result["items"]) == 1

    @pytest.mark.asyncio
    async def test_items_have_year_from_title(self, real_analyzer: ContentAnalyzerAgent) -> None:
        result = await real_analyzer.run({"user_id": 0, "item_ids": [0]})
        item = result["items"][0]
        assert item["year"] == 2000

    @pytest.mark.asyncio
    async def test_items_have_parsed_genres(self, real_analyzer: ContentAnalyzerAgent) -> None:
        result = await real_analyzer.run({"user_id": 0, "item_ids": [0]})
        genres = result["items"][0]["genres"]
        assert "Action" in genres
        assert "Comedy" in genres

    @pytest.mark.asyncio
    async def test_items_have_embedding(self, real_analyzer: ContentAnalyzerAgent) -> None:
        result = await real_analyzer.run({"user_id": 0, "item_ids": [1]})
        assert len(result["items"][0]["embedding"]) == EMBEDDING_DIM


# ── Mock fallback still works ────────────────────────────────────────


class TestMockFallback:
    @pytest.mark.asyncio
    async def test_profiler_mock_mode(self, store: EmbeddingStore) -> None:
        profiler = UserProfilerAgent(embedding_store=store)
        assert profiler.has_real_data is False
        result = await profiler.run({"user_id": 1})
        assert result["rating_count"] == 5  # mock user 1 has 5 ratings

    @pytest.mark.asyncio
    async def test_analyzer_mock_mode(self, store: EmbeddingStore) -> None:
        analyzer = ContentAnalyzerAgent(embedding_store=store)
        assert analyzer.has_real_data is False
        result = await analyzer.run({"user_id": 1})
        assert len(result["items"]) == 10  # full mock catalog
