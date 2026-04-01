"""Tests for all API routes."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.api import dependencies as deps
from src.data.loader import DataSplits
from src.models.ncf import NeuralCollaborativeFiltering


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NUM_USERS = 3
_NUM_ITEMS = 5


@pytest.fixture(autouse=True)
def _wire_dependencies() -> None:  # noqa: ANN202
    """Patch module-level singletons so tests run without real infra."""
    model = NeuralCollaborativeFiltering(
        num_users=_NUM_USERS, num_items=_NUM_ITEMS, embedding_dim=32,
    )
    model.eval()

    ratings = pd.DataFrame({
        "userId": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "movieId": [0, 1, 2, 0, 1, 3, 2, 3, 4],
        "rating": [0.9, 0.7, 0.3, 0.5, 0.8, 0.6, 0.4, 0.9, 0.2],
        "timestamp": list(range(9)),
    })
    movies = pd.DataFrame({
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
            "Drama",
            "Sci-Fi|Thriller",
            "Romance",
            "Horror",
        ],
    })

    splits = DataSplits(
        train=ratings.iloc[:6],
        val=ratings.iloc[6:8],
        test=ratings.iloc[8:],
        movies=movies,
        user_map={0: 0, 1: 1, 2: 2},
        movie_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    )

    deps._model = model
    deps._splits = splits

    from src.models.embeddings import EmbeddingStore

    store = EmbeddingStore(embedding_dim=32)
    store.load_from_model(model)
    deps._embedding_store = store

    from src.agents.content_analyzer import ContentAnalyzerAgent
    from src.agents.orchestrator import OrchestratorAgent
    from src.agents.recsys_engine import RecsysEngineAgent
    from src.agents.user_profiler import UserProfilerAgent

    all_ratings = pd.concat([splits.train, splits.val, splits.test], ignore_index=True)
    profiler = UserProfilerAgent(
        embedding_store=store, ratings_df=all_ratings, movies_df=movies,
    )
    analyzer = ContentAnalyzerAgent(embedding_store=store, movies_df=movies)
    engine = RecsysEngineAgent(model=model, embedding_store=store)
    deps._orchestrator = OrchestratorAgent(
        user_profiler=profiler,
        content_analyzer=analyzer,
        recsys_engine=engine,
    )

    yield

    # Reset
    deps._model = None
    deps._splits = None
    deps._embedding_store = None
    deps._orchestrator = None


@pytest.fixture
def transport() -> ASGITransport:
    """Build the ASGI transport without triggering the lifespan."""
    from src.main import app

    return ASGITransport(app=app)


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------


class TestPostRecommend:
    @pytest.mark.asyncio
    async def test_returns_recommendations(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/v1/recommend", json={"user_id": 0, "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == 0
        assert len(body["recommendations"]) <= 3
        assert body["model"] == "ncf"

    @pytest.mark.asyncio
    async def test_recommendations_have_scores(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/v1/recommend", json={"user_id": 0})
        items = resp.json()["recommendations"]
        assert all("score" in item for item in items)
        # Scores should be in [0, 1] (sigmoid output)
        assert all(0.0 <= item["score"] <= 1.0 for item in items)

    @pytest.mark.asyncio
    async def test_validation_error(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/v1/recommend", json={"user_id": 0, "top_k": 0})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /recommend/{user_id}
# ---------------------------------------------------------------------------


class TestGetRecommend:
    @pytest.mark.asyncio
    async def test_defaults(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/recommend/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == 1

    @pytest.mark.asyncio
    async def test_custom_top_k(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/recommend/0?top_k=2")
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) <= 2


# ---------------------------------------------------------------------------
# GET /user/{user_id}/history
# ---------------------------------------------------------------------------


class TestUserHistory:
    @pytest.mark.asyncio
    async def test_known_user(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/user/0/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == 0
        assert body["rating_count"] == 3
        assert len(body["ratings"]) == 3

    @pytest.mark.asyncio
    async def test_ratings_have_metadata(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/user/0/history")
        first = resp.json()["ratings"][0]
        assert "title" in first
        assert "genres" in first

    @pytest.mark.asyncio
    async def test_unknown_user_returns_404(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/user/999/history")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_no_data_returns_503(self, transport: ASGITransport) -> None:
        deps._splits = None
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/user/0/history")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /movies/search
# ---------------------------------------------------------------------------


class TestMovieSearch:
    @pytest.mark.asyncio
    async def test_search_by_title(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/movies/search", params={"q": "Alpha"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "Alpha"
        assert body["total"] >= 1
        assert body["results"][0]["title"] == "Alpha (2000)"

    @pytest.mark.asyncio
    async def test_case_insensitive(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/movies/search", params={"q": "beta"})
        assert resp.json()["total"] >= 1

    @pytest.mark.asyncio
    async def test_no_results(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/movies/search", params={"q": "zzz_no_match"})
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_limit(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/movies/search", params={"q": "a", "limit": 2})
        assert len(resp.json()["results"]) <= 2

    @pytest.mark.asyncio
    async def test_result_has_year(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/movies/search", params={"q": "Alpha"})
        assert resp.json()["results"][0]["year"] == 2000

    @pytest.mark.asyncio
    async def test_missing_query_returns_422(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/movies/search")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_no_data_returns_503(self, transport: ASGITransport) -> None:
        deps._splits = None
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/movies/search", params={"q": "Alpha"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    @pytest.mark.asyncio
    async def test_healthy(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["data_loaded"] is True
        assert body["num_users"] == _NUM_USERS
        assert body["num_movies"] == _NUM_ITEMS

    @pytest.mark.asyncio
    async def test_agents_listed(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/health")
        agents = resp.json()["agents"]
        names = {a["name"] for a in agents}
        assert "user_profiler" in names
        assert "content_analyzer" in names
        assert "recsys_engine" in names

    @pytest.mark.asyncio
    async def test_degraded_without_model(self, transport: ASGITransport) -> None:
        deps._model = None
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/health")
        assert resp.json()["status"] == "degraded"
        assert resp.json()["model_loaded"] is False

    @pytest.mark.asyncio
    async def test_degraded_without_data(self, transport: ASGITransport) -> None:
        deps._splits = None
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/health")
        assert resp.json()["status"] == "degraded"
        assert resp.json()["data_loaded"] is False


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    @pytest.mark.asyncio
    async def test_returns_prometheus_text(self, transport: ASGITransport) -> None:
        # Make a request first so there's something to report.
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            await c.get("/api/v1/health")
            resp = await c.get("/api/v1/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "agentrec_requests_total" in resp.text

    @pytest.mark.asyncio
    async def test_contains_model_inference(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/metrics")
        assert "agentrec_model_inference_seconds" in resp.text


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class TestMiddleware:
    @pytest.mark.asyncio
    async def test_request_id_generated(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/health")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0

    @pytest.mark.asyncio
    async def test_request_id_preserved(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(
                "/api/v1/health",
                headers={"X-Request-ID": "test-id-123"},
            )
        assert resp.headers["x-request-id"] == "test-id-123"

    @pytest.mark.asyncio
    async def test_cors_headers(self, transport: ASGITransport) -> None:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.options(
                "/api/v1/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert "access-control-allow-origin" in resp.headers
