"""Shared pytest fixtures for AgentRec tests."""

import pytest

from src.models.embeddings import EmbeddingStore
from src.agents.user_profiler import UserProfilerAgent
from src.agents.content_analyzer import ContentAnalyzerAgent
from src.agents.recsys_engine import RecsysEngineAgent
from src.agents.orchestrator import OrchestratorAgent


@pytest.fixture
def embedding_store() -> EmbeddingStore:
    """Return a fresh EmbeddingStore for test isolation."""
    return EmbeddingStore(embedding_dim=64)


@pytest.fixture
def user_profiler(embedding_store: EmbeddingStore) -> UserProfilerAgent:
    """Return a UserProfilerAgent instance."""
    return UserProfilerAgent(embedding_store=embedding_store)


@pytest.fixture
def content_analyzer(embedding_store: EmbeddingStore) -> ContentAnalyzerAgent:
    """Return a ContentAnalyzerAgent instance."""
    return ContentAnalyzerAgent(embedding_store=embedding_store)


@pytest.fixture
def recsys_engine(embedding_store: EmbeddingStore) -> RecsysEngineAgent:
    """Return a RecsysEngineAgent instance."""
    return RecsysEngineAgent(embedding_store=embedding_store)


@pytest.fixture
def orchestrator(
    user_profiler: UserProfilerAgent,
    content_analyzer: ContentAnalyzerAgent,
    recsys_engine: RecsysEngineAgent,
) -> OrchestratorAgent:
    """Return a fully-wired OrchestratorAgent instance."""
    return OrchestratorAgent(
        user_profiler=user_profiler,
        content_analyzer=content_analyzer,
        recsys_engine=recsys_engine,
    )
