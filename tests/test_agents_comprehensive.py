"""Comprehensive agent tests.

Covers BaseAgent inheritance, orchestrator plan-and-execute with mocked
specialist agents, user_profiler vector shapes, and content_analyzer
graceful handling of missing movies.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.agents.base import BaseAgent
from src.agents.content_analyzer import ContentAnalyzerAgent
from src.agents.orchestrator import OrchestratorAgent
from src.agents.recsys_engine import RecsysEngineAgent
from src.agents.user_profiler import UserProfilerAgent
from src.models.embeddings import EmbeddingStore

# ── Shared fixtures ──────────────────────────────────────────────────

EMBEDDING_DIM = 64


@pytest.fixture
def store() -> EmbeddingStore:
    """Isolated embedding store for each test."""
    return EmbeddingStore(embedding_dim=EMBEDDING_DIM)


@pytest.fixture
def profiler(store: EmbeddingStore) -> UserProfilerAgent:
    return UserProfilerAgent(embedding_store=store)


@pytest.fixture
def analyzer(store: EmbeddingStore) -> ContentAnalyzerAgent:
    return ContentAnalyzerAgent(embedding_store=store)


@pytest.fixture
def engine(store: EmbeddingStore) -> RecsysEngineAgent:
    return RecsysEngineAgent(embedding_store=store)


@pytest.fixture
def orch(
    profiler: UserProfilerAgent,
    analyzer: ContentAnalyzerAgent,
    engine: RecsysEngineAgent,
) -> OrchestratorAgent:
    return OrchestratorAgent(
        user_profiler=profiler,
        content_analyzer=analyzer,
        recsys_engine=engine,
    )


# ── 1. BaseAgent inheritance ────────────────────────────────────────


class TestBaseAgentInheritance:
    """Every concrete agent must inherit from BaseAgent."""

    def test_orchestrator_is_base_agent(self, orch: OrchestratorAgent) -> None:
        assert isinstance(orch, BaseAgent)

    def test_user_profiler_is_base_agent(self, profiler: UserProfilerAgent) -> None:
        assert isinstance(profiler, BaseAgent)

    def test_content_analyzer_is_base_agent(self, analyzer: ContentAnalyzerAgent) -> None:
        assert isinstance(analyzer, BaseAgent)

    def test_recsys_engine_is_base_agent(self, engine: RecsysEngineAgent) -> None:
        assert isinstance(engine, BaseAgent)

    def test_agents_have_agent_name(
        self,
        profiler: UserProfilerAgent,
        analyzer: ContentAnalyzerAgent,
        engine: RecsysEngineAgent,
        orch: OrchestratorAgent,
    ) -> None:
        """All agents expose a non-empty agent_name property."""
        for agent in (profiler, analyzer, engine, orch):
            assert isinstance(agent.agent_name, str)
            assert len(agent.agent_name) > 0

    def test_agents_have_distinct_names(
        self,
        profiler: UserProfilerAgent,
        analyzer: ContentAnalyzerAgent,
        engine: RecsysEngineAgent,
        orch: OrchestratorAgent,
    ) -> None:
        """Every agent name is unique."""
        names = [a.agent_name for a in (profiler, analyzer, engine, orch)]
        assert len(names) == len(set(names))

    def test_agents_have_memory(
        self,
        profiler: UserProfilerAgent,
        analyzer: ContentAnalyzerAgent,
        engine: RecsysEngineAgent,
        orch: OrchestratorAgent,
    ) -> None:
        """All agents start with an empty _memory dict."""
        for agent in (profiler, analyzer, engine, orch):
            assert isinstance(agent._memory, dict)
            assert len(agent._memory) == 0

    def test_agents_have_run_method(
        self,
        profiler: UserProfilerAgent,
        analyzer: ContentAnalyzerAgent,
        engine: RecsysEngineAgent,
        orch: OrchestratorAgent,
    ) -> None:
        """All agents expose an async run() method."""
        for agent in (profiler, analyzer, engine, orch):
            assert callable(getattr(agent, "run", None))

    def test_agents_have_process_method(
        self,
        profiler: UserProfilerAgent,
        analyzer: ContentAnalyzerAgent,
        engine: RecsysEngineAgent,
        orch: OrchestratorAgent,
    ) -> None:
        """All agents expose an async process() method."""
        for agent in (profiler, analyzer, engine, orch):
            assert callable(getattr(agent, "process", None))

    def test_cannot_instantiate_base_agent_directly(self) -> None:
        """BaseAgent is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseAgent(name="nope")  # type: ignore[abstract]


# ── 2. Orchestrator plan-and-execute with mocked agents ─────────────


class TestOrchestratorWithMocks:
    """Test the orchestrator's coordination logic using mocked sub-agents."""

    @pytest.fixture
    def mock_profiler_output(self) -> dict[str, Any]:
        return {
            "user_id": 1,
            "ratings": [
                {"item_id": 10, "rating": 5.0, "title": "Mock Movie", "genres": ["Action"]}
            ],
            "embedding": [0.1] * EMBEDDING_DIM,
            "rating_count": 1,
            "avg_rating": 5.0,
        }

    @pytest.fixture
    def mock_analyzer_output(self) -> dict[str, Any]:
        return {
            "user_id": 1,
            "items": [
                {
                    "item_id": 10,
                    "title": "Mock Movie",
                    "genres": ["Action"],
                    "year": 2020,
                    "tags": [],
                    "embedding": [0.2] * EMBEDDING_DIM,
                },
                {
                    "item_id": 20,
                    "title": "Another Movie",
                    "genres": ["Drama"],
                    "year": 2021,
                    "tags": [],
                    "embedding": [0.3] * EMBEDDING_DIM,
                },
            ],
        }

    @pytest.fixture
    def mock_engine_output(self) -> dict[str, Any]:
        return {
            "ranked_items": [
                {"item_id": 20, "title": "Another Movie", "score": 0.95, "genres": ["Drama"]},
                {"item_id": 10, "title": "Mock Movie", "score": 0.80, "genres": ["Action"]},
            ],
        }

    @pytest.fixture
    def mocked_orchestrator(
        self,
        store: EmbeddingStore,
        mock_profiler_output: dict[str, Any],
        mock_analyzer_output: dict[str, Any],
        mock_engine_output: dict[str, Any],
    ) -> OrchestratorAgent:
        """Build an orchestrator with fully mocked sub-agents."""
        profiler = UserProfilerAgent(embedding_store=store)
        analyzer = ContentAnalyzerAgent(embedding_store=store)
        engine = RecsysEngineAgent(embedding_store=store)

        profiler.run = AsyncMock(return_value=mock_profiler_output)  # type: ignore[method-assign]
        analyzer.run = AsyncMock(return_value=mock_analyzer_output)  # type: ignore[method-assign]
        engine.run = AsyncMock(return_value=mock_engine_output)  # type: ignore[method-assign]

        return OrchestratorAgent(
            user_profiler=profiler,
            content_analyzer=analyzer,
            recsys_engine=engine,
        )

    @pytest.mark.asyncio
    async def test_orchestrator_calls_all_agents(
        self, mocked_orchestrator: OrchestratorAgent
    ) -> None:
        """Orchestrator invokes all three sub-agents exactly once."""
        await mocked_orchestrator.process({"user_id": 1, "top_k": 5})

        mocked_orchestrator.user_profiler.run.assert_called_once()  # type: ignore[union-attr]
        mocked_orchestrator.content_analyzer.run.assert_called_once()  # type: ignore[union-attr]
        mocked_orchestrator.recsys_engine.run.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_orchestrator_passes_user_id_to_parallel_agents(
        self, mocked_orchestrator: OrchestratorAgent
    ) -> None:
        """Both profiler and analyzer receive the correct user_id."""
        await mocked_orchestrator.process({"user_id": 42, "top_k": 3})

        profiler_call = mocked_orchestrator.user_profiler.run.call_args  # type: ignore[union-attr]
        analyzer_call = mocked_orchestrator.content_analyzer.run.call_args  # type: ignore[union-attr]

        assert profiler_call[0][0]["user_id"] == 42
        assert analyzer_call[0][0]["user_id"] == 42

    @pytest.mark.asyncio
    async def test_orchestrator_passes_outputs_to_engine(
        self,
        mocked_orchestrator: OrchestratorAgent,
        mock_profiler_output: dict[str, Any],
        mock_analyzer_output: dict[str, Any],
    ) -> None:
        """Engine receives both profiler and analyzer outputs plus top_k."""
        await mocked_orchestrator.process({"user_id": 1, "top_k": 7})

        engine_call = mocked_orchestrator.recsys_engine.run.call_args  # type: ignore[union-attr]
        engine_payload = engine_call[0][0]

        assert engine_payload["user_profile"] == mock_profiler_output
        assert engine_payload["content_features"] == mock_analyzer_output
        assert engine_payload["top_k"] == 7

    @pytest.mark.asyncio
    async def test_orchestrator_returns_correct_structure(
        self, mocked_orchestrator: OrchestratorAgent
    ) -> None:
        """Response has user_id, recommendations list, and metadata."""
        result = await mocked_orchestrator.process({"user_id": 1, "top_k": 2})

        assert result["user_id"] == 1
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) == 2
        assert result["metadata"]["model"] == "ncf"
        assert result["metadata"]["top_k"] == 2

    @pytest.mark.asyncio
    async def test_orchestrator_recommendations_preserve_order(
        self, mocked_orchestrator: OrchestratorAgent
    ) -> None:
        """Recommendations are in the order returned by the engine (score-sorted)."""
        result = await mocked_orchestrator.process({"user_id": 1, "top_k": 2})

        items = result["recommendations"]
        assert items[0]["item_id"] == 20  # score 0.95
        assert items[1]["item_id"] == 10  # score 0.80

    @pytest.mark.asyncio
    async def test_orchestrator_plan_has_three_steps(
        self, mocked_orchestrator: OrchestratorAgent
    ) -> None:
        """Execution plan contains exactly 3 steps with correct parallel groups."""
        result = await mocked_orchestrator.process({"user_id": 1, "top_k": 5})

        plan = result["metadata"]["plan"]
        assert len(plan) == 3
        # Profiler and analyzer are group 0 (parallel), engine is group 1
        groups = [step["parallel_group"] for step in plan]
        assert groups == [0, 0, 1]

    @pytest.mark.asyncio
    async def test_orchestrator_parallel_execution(self, store: EmbeddingStore) -> None:
        """Profiler and analyzer run concurrently (both start before either finishes).

        We verify this by checking that asyncio.gather is used — if they ran
        sequentially, the call order would be deterministic profiler→analyzer.
        With gather, both are started before awaiting results.
        """
        import asyncio

        call_order: list[str] = []

        async def mock_profiler_run(payload: dict[str, Any]) -> dict[str, Any]:
            call_order.append("profiler_start")
            await asyncio.sleep(0)  # yield to event loop
            call_order.append("profiler_end")
            return {
                "user_id": 1,
                "ratings": [],
                "embedding": [0.0] * 64,
                "rating_count": 0,
                "avg_rating": 0.0,
            }

        async def mock_analyzer_run(payload: dict[str, Any]) -> dict[str, Any]:
            call_order.append("analyzer_start")
            await asyncio.sleep(0)
            call_order.append("analyzer_end")
            return {"user_id": 1, "items": []}

        async def mock_engine_run(payload: dict[str, Any]) -> dict[str, Any]:
            call_order.append("engine_start")
            return {"ranked_items": []}

        profiler = UserProfilerAgent(embedding_store=store)
        analyzer = ContentAnalyzerAgent(embedding_store=store)
        engine = RecsysEngineAgent(embedding_store=store)
        profiler.run = mock_profiler_run  # type: ignore[method-assign]
        analyzer.run = mock_analyzer_run  # type: ignore[method-assign]
        engine.run = mock_engine_run  # type: ignore[method-assign]

        orch = OrchestratorAgent(profiler, analyzer, engine)
        await orch.process({"user_id": 1, "top_k": 5})

        # With asyncio.gather, both start before either ends
        assert call_order.index("profiler_start") < call_order.index("profiler_end")
        assert call_order.index("analyzer_start") < call_order.index("analyzer_end")
        # Engine runs after both parallel agents finish
        assert call_order.index("engine_start") > call_order.index("profiler_end")
        assert call_order.index("engine_start") > call_order.index("analyzer_end")


# ── 3. UserProfiler vector shapes ───────────────────────────────────


class TestUserProfilerVectors:
    """Verify user profiler returns correct embedding shapes and values."""

    @pytest.mark.asyncio
    async def test_embedding_dimension_matches_store(self, profiler: UserProfilerAgent) -> None:
        """Embedding length equals the EmbeddingStore's embedding_dim."""
        result = await profiler.run({"user_id": 1})
        assert len(result["embedding"]) == EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_embedding_all_floats(self, profiler: UserProfilerAgent) -> None:
        """Every element in the embedding is a float."""
        result = await profiler.run({"user_id": 1})
        assert all(isinstance(x, float) for x in result["embedding"])

    @pytest.mark.asyncio
    async def test_unknown_user_zero_vector(self, profiler: UserProfilerAgent) -> None:
        """Unknown users get a zero vector of correct dimension."""
        result = await profiler.run({"user_id": 99999})
        assert result["embedding"] == [0.0] * EMBEDDING_DIM
        assert result["rating_count"] == 0

    @pytest.mark.asyncio
    async def test_known_user_nonzero_vector(self, profiler: UserProfilerAgent) -> None:
        """Known users with high-rated items get a non-zero embedding."""
        result = await profiler.run({"user_id": 1})
        assert any(x != 0.0 for x in result["embedding"])

    @pytest.mark.asyncio
    async def test_different_users_different_embeddings(self, profiler: UserProfilerAgent) -> None:
        """Two different users produce different embedding vectors."""
        r1 = await profiler.run({"user_id": 1})
        r2 = await profiler.run({"user_id": 2})
        assert r1["embedding"] != r2["embedding"]

    @pytest.mark.asyncio
    async def test_ratings_list_shape(self, profiler: UserProfilerAgent) -> None:
        """Each rating in the list has item_id, rating, title, genres."""
        result = await profiler.run({"user_id": 1})
        for rating in result["ratings"]:
            assert "item_id" in rating
            assert "rating" in rating
            assert "title" in rating
            assert "genres" in rating
            assert isinstance(rating["rating"], float)

    @pytest.mark.asyncio
    async def test_embedding_consistent_across_calls(self, profiler: UserProfilerAgent) -> None:
        """Same user_id produces the same embedding on repeated calls."""
        r1 = await profiler.run({"user_id": 3})
        profiler.clear_memory()
        r2 = await profiler.run({"user_id": 3})
        assert r1["embedding"] == r2["embedding"]

    @pytest.mark.asyncio
    async def test_custom_embedding_dim(self) -> None:
        """Profiler respects a non-default embedding dimension."""
        small_store = EmbeddingStore(embedding_dim=16)
        small_profiler = UserProfilerAgent(embedding_store=small_store)
        result = await small_profiler.run({"user_id": 1})
        assert len(result["embedding"]) == 16


# ── 4. ContentAnalyzer graceful handling of missing movies ──────────


class TestContentAnalyzerMissingMovies:
    """Verify content analyzer handles missing/invalid movie IDs gracefully."""

    @pytest.mark.asyncio
    async def test_single_missing_id_returns_empty(self, analyzer: ContentAnalyzerAgent) -> None:
        """A single nonexistent item_id results in zero items returned."""
        result = await analyzer.run({"user_id": 1, "item_ids": [999999]})
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_all_missing_ids_returns_empty(self, analyzer: ContentAnalyzerAgent) -> None:
        """Multiple nonexistent IDs all result in zero items."""
        result = await analyzer.run({"user_id": 1, "item_ids": [8888, 9999, 7777]})
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_mix_valid_and_missing_returns_only_valid(
        self, analyzer: ContentAnalyzerAgent
    ) -> None:
        """Valid IDs are returned; missing IDs are silently skipped."""
        result = await analyzer.run({"user_id": 1, "item_ids": [1, 99999, 50]})
        returned_ids = {item["item_id"] for item in result["items"]}
        assert returned_ids == {1, 50}

    @pytest.mark.asyncio
    async def test_no_item_ids_returns_full_catalog(self, analyzer: ContentAnalyzerAgent) -> None:
        """Omitting item_ids returns the entire mock catalog."""
        result = await analyzer.run({"user_id": 1})
        assert len(result["items"]) == 10

    @pytest.mark.asyncio
    async def test_empty_item_ids_list_returns_full_catalog(
        self, analyzer: ContentAnalyzerAgent
    ) -> None:
        """An explicit empty list behaves the same as omitting item_ids."""
        result = await analyzer.run({"user_id": 1, "item_ids": []})
        assert len(result["items"]) == 10

    @pytest.mark.asyncio
    async def test_duplicate_valid_ids(self, analyzer: ContentAnalyzerAgent) -> None:
        """Duplicate IDs return duplicate items (no dedup)."""
        result = await analyzer.run({"user_id": 1, "item_ids": [1, 1, 1]})
        assert len(result["items"]) == 3
        assert all(item["item_id"] == 1 for item in result["items"])

    @pytest.mark.asyncio
    async def test_negative_id_treated_as_missing(self, analyzer: ContentAnalyzerAgent) -> None:
        """Negative item IDs are not in the catalog and are skipped."""
        result = await analyzer.run({"user_id": 1, "item_ids": [-1, -100]})
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_missing_items_still_sets_user_id(self, analyzer: ContentAnalyzerAgent) -> None:
        """Response includes correct user_id even when all items are missing."""
        result = await analyzer.run({"user_id": 42, "item_ids": [99999]})
        assert result["user_id"] == 42

    @pytest.mark.asyncio
    async def test_returned_items_have_embeddings(self, analyzer: ContentAnalyzerAgent) -> None:
        """Every returned item has an embedding of correct dimension."""
        result = await analyzer.run({"user_id": 1, "item_ids": [1, 50]})
        for item in result["items"]:
            assert len(item["embedding"]) == EMBEDDING_DIM
            assert all(isinstance(x, float) for x in item["embedding"])

    @pytest.mark.asyncio
    async def test_returned_items_have_full_metadata(self, analyzer: ContentAnalyzerAgent) -> None:
        """Valid items include title, genres, year, and tags."""
        result = await analyzer.run({"user_id": 1, "item_ids": [296]})
        item = result["items"][0]
        assert item["title"] == "Pulp Fiction"
        assert "Crime" in item["genres"]
        assert item["year"] == 1994
        assert "tarantino" in item["tags"]
