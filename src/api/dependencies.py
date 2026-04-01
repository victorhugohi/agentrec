"""FastAPI dependency injection providers.

Provides singletons for the NCF model, data splits, embedding store,
database pool, and fully-wired orchestrator agent.  All are lazily
initialised on first access and cached for the process lifetime.
"""

from __future__ import annotations

import logging
from pathlib import Path

import asyncpg

from src.agents.content_analyzer import ContentAnalyzerAgent
from src.agents.orchestrator import OrchestratorAgent
from src.agents.recsys_engine import RecsysEngineAgent
from src.agents.user_profiler import UserProfilerAgent
from src.config import get_settings
from src.data.database import get_pool
from src.data.loader import DataSplits, MovieLensLoader
from src.models.embeddings import EmbeddingStore
from src.models.ncf import NeuralCollaborativeFiltering

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (populated at startup or lazily)
# ---------------------------------------------------------------------------

_model: NeuralCollaborativeFiltering | None = None
_splits: DataSplits | None = None
_embedding_store: EmbeddingStore | None = None
_orchestrator: OrchestratorAgent | None = None


# ---------------------------------------------------------------------------
# Startup / Teardown
# ---------------------------------------------------------------------------


def load_model() -> NeuralCollaborativeFiltering | None:
    """Load the trained NCF model from the configured checkpoint.

    Returns:
        The loaded model, or ``None`` if the checkpoint doesn't exist.
    """
    global _model
    settings = get_settings()
    path = Path(settings.model_path)
    if not path.exists():
        logger.warning("Model checkpoint not found at %s — running without model", path)
        _model = None
        return None

    _model = NeuralCollaborativeFiltering.load_pretrained(path)
    logger.info("Loaded NCF model from %s", path)
    return _model


def load_data() -> DataSplits | None:
    """Load processed MovieLens data splits.

    Returns:
        The data splits, or ``None`` if processed data is missing.
    """
    global _splits
    settings = get_settings()
    loader = MovieLensLoader(variant=settings.movielens_variant, data_dir=settings.data_dir)
    _splits = loader.load_processed()
    if _splits is not None:
        logger.info(
            "Loaded data: %d users, %d movies",
            _splits.num_users, _splits.num_movies,
        )
    else:
        logger.warning("No processed data found in %s", settings.data_dir)
    return _splits


def build_orchestrator() -> OrchestratorAgent:
    """Build the orchestrator with all sub-agents wired to DB, data, and model."""
    global _orchestrator, _embedding_store

    settings = get_settings()
    _embedding_store = EmbeddingStore(embedding_dim=settings.embedding_dim)

    if _model is not None:
        _embedding_store.load_from_model(_model)

    db_pool = get_pool()

    # Prefer DB, fall back to DataFrame, then mock.
    if db_pool is not None:
        user_profiler = UserProfilerAgent(
            embedding_store=_embedding_store,
            db_pool=db_pool,
        )
        content_analyzer = ContentAnalyzerAgent(
            embedding_store=_embedding_store,
            db_pool=db_pool,
        )
    elif _splits is not None:
        import pandas as pd

        all_ratings = pd.concat([_splits.train, _splits.val, _splits.test], ignore_index=True)
        user_profiler = UserProfilerAgent(
            embedding_store=_embedding_store,
            ratings_df=all_ratings,
            movies_df=_splits.movies,
        )
        content_analyzer = ContentAnalyzerAgent(
            embedding_store=_embedding_store,
            movies_df=_splits.movies,
        )
    else:
        user_profiler = UserProfilerAgent(embedding_store=_embedding_store)
        content_analyzer = ContentAnalyzerAgent(embedding_store=_embedding_store)

    recsys_engine = RecsysEngineAgent(
        model=_model,
        embedding_store=_embedding_store,
    )

    _orchestrator = OrchestratorAgent(
        user_profiler=user_profiler,
        content_analyzer=content_analyzer,
        recsys_engine=recsys_engine,
    )
    logger.info("Orchestrator built (model=%s, db=%s, data=%s)",
                _model is not None, db_pool is not None, _splits is not None)
    return _orchestrator


# ---------------------------------------------------------------------------
# FastAPI dependency callables
# ---------------------------------------------------------------------------


def get_model() -> NeuralCollaborativeFiltering | None:
    """Return the loaded NCF model (may be None)."""
    return _model


def get_splits() -> DataSplits | None:
    """Return the loaded data splits (may be None)."""
    return _splits


def get_db_pool() -> asyncpg.Pool | None:
    """Return the asyncpg connection pool (may be None)."""
    return get_pool()


def get_embedding_store() -> EmbeddingStore | None:
    """Return the shared embedding store."""
    return _embedding_store


def get_orchestrator() -> OrchestratorAgent:
    """Return the cached orchestrator agent.

    Returns:
        A fully-wired OrchestratorAgent instance.

    Raises:
        RuntimeError: If the orchestrator hasn't been built yet.
    """
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialized. App startup incomplete.")
    return _orchestrator
