"""Shared embedding store for users and items.

Provides a single access point for embedding vectors used by agents
(user_profiler, content_analyzer) and the NCF model.  When a trained
NCF model is available, embeddings are extracted directly from the model's
GMF embedding tables.  Otherwise, a fallback random embedding is generated
deterministically so behaviour is reproducible.

Usage::

    store = EmbeddingStore(embedding_dim=64)

    # After training or loading a model:
    store.load_from_model(model)

    user_vec = store.user_embedding(user_id=42)
    item_vec = store.item_embedding(item_id=318)
"""

from __future__ import annotations

import logging

import torch

from src.models.ncf import NeuralCollaborativeFiltering

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Centralised embedding lookup shared across agents and the NCF model.

    The store operates in two modes:

    1. **Model-backed** (after :meth:`load_from_model`): embeddings come
       straight from the NCF model's trained GMF embedding tables.  This is
       the production path.
    2. **Fallback**: when no model is loaded, a deterministic random vector
       is generated per ID so that the rest of the pipeline can run
       (useful for tests and development).

    Attributes:
        embedding_dim: Length of every embedding vector.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        """Initialise the store.

        Args:
            embedding_dim: Dimensionality of embedding vectors.
        """
        self.embedding_dim = embedding_dim
        self._model: NeuralCollaborativeFiltering | None = None
        self._user_cache: dict[int, list[float]] = {}
        self._item_cache: dict[int, list[float]] = {}

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_from_model(self, model: NeuralCollaborativeFiltering) -> None:
        """Attach a trained NCF model as the embedding source.

        Clears any cached fallback embeddings so subsequent lookups
        hit the model's tables instead.

        Args:
            model: A trained :class:`NeuralCollaborativeFiltering` instance.
        """
        self._model = model
        self._user_cache.clear()
        self._item_cache.clear()
        logger.info(
            "EmbeddingStore: loaded model (users=%d, items=%d, dim=%d)",
            model.num_users,
            model.num_items,
            model.embedding_dim,
        )

    @property
    def has_model(self) -> bool:
        """Return whether a trained model is currently attached."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Embedding lookups
    # ------------------------------------------------------------------

    def user_embedding(self, user_id: int) -> list[float]:
        """Return the embedding vector for a user.

        Uses the NCF model if loaded, otherwise falls back to a
        deterministic random vector.  Results are cached.

        Args:
            user_id: Integer user identifier.

        Returns:
            A list of floats with length ``embedding_dim``.
        """
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        if self._model is not None and 0 <= user_id < self._model.num_users:
            vec = self._model.get_user_embedding(user_id).tolist()
        else:
            vec = self._fallback_embedding(seed=user_id)

        self._user_cache[user_id] = vec
        return vec

    def item_embedding(self, item_id: int) -> list[float]:
        """Return the embedding vector for an item.

        Uses the NCF model if loaded, otherwise falls back to a
        deterministic random vector.  Results are cached.

        Args:
            item_id: Integer item identifier.

        Returns:
            A list of floats with length ``embedding_dim``.
        """
        if item_id in self._item_cache:
            return self._item_cache[item_id]

        if self._model is not None and 0 <= item_id < self._model.num_items:
            vec = self._model.get_item_embedding(item_id).tolist()
        else:
            vec = self._fallback_embedding(seed=item_id + 1_000_000)

        self._item_cache[item_id] = vec
        return vec

    def user_embeddings_batch(self, user_ids: list[int]) -> list[list[float]]:
        """Return embeddings for multiple users at once.

        Args:
            user_ids: List of user identifiers.

        Returns:
            A list of embedding vectors, one per user, in the same order.
        """
        return [self.user_embedding(uid) for uid in user_ids]

    def item_embeddings_batch(self, item_ids: list[int]) -> list[list[float]]:
        """Return embeddings for multiple items at once.

        Args:
            item_ids: List of item identifiers.

        Returns:
            A list of embedding vectors, one per item, in the same order.
        """
        return [self.item_embedding(iid) for iid in item_ids]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fallback_embedding(self, seed: int) -> list[float]:
        """Generate a deterministic pseudo-random embedding.

        Uses a seeded PyTorch generator so the same ID always produces
        the same vector, making tests reproducible even without a model.

        Args:
            seed: Integer seed (typically the entity ID).

        Returns:
            A list of floats with length ``embedding_dim``.
        """
        gen = torch.Generator().manual_seed(seed)
        vec = torch.randn(self.embedding_dim, generator=gen)
        # L2-normalise so magnitudes are comparable to trained embeddings.
        vec = vec / (vec.norm() + 1e-8)
        return vec.tolist()

    def clear_cache(self) -> None:
        """Flush the embedding cache, forcing re-computation on next lookup."""
        self._user_cache.clear()
        self._item_cache.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: EmbeddingStore | None = None


def get_embedding_store(embedding_dim: int = 64) -> EmbeddingStore:
    """Return the global singleton :class:`EmbeddingStore`.

    Creates the store on first call.  Subsequent calls return the same
    instance.

    Args:
        embedding_dim: Dimensionality (only used on first call).

    Returns:
        The shared :class:`EmbeddingStore` instance.
    """
    global _store
    if _store is None:
        _store = EmbeddingStore(embedding_dim=embedding_dim)
    return _store
