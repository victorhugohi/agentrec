"""Tests for the shared EmbeddingStore."""

import pytest

from src.models.embeddings import EmbeddingStore
from src.models.ncf import NeuralCollaborativeFiltering


@pytest.fixture
def store() -> EmbeddingStore:
    """Return a fresh EmbeddingStore (no model attached)."""
    return EmbeddingStore(embedding_dim=16)


@pytest.fixture
def model() -> NeuralCollaborativeFiltering:
    """Return a small NCF model."""
    return NeuralCollaborativeFiltering(num_users=50, num_items=30, embedding_dim=16)


def test_fallback_embedding_dim(store: EmbeddingStore) -> None:
    """Fallback embeddings have the correct dimensionality."""
    vec = store.user_embedding(0)
    assert len(vec) == 16


def test_fallback_deterministic(store: EmbeddingStore) -> None:
    """Same ID always produces the same fallback vector."""
    a = store.user_embedding(42)
    store.clear_cache()
    b = store.user_embedding(42)
    assert a == b


def test_user_and_item_differ(store: EmbeddingStore) -> None:
    """User and item embeddings for the same numeric ID differ."""
    user_vec = store.user_embedding(1)
    item_vec = store.item_embedding(1)
    assert user_vec != item_vec


def test_has_model_false_by_default(store: EmbeddingStore) -> None:
    """has_model is False before loading a model."""
    assert store.has_model is False


def test_load_from_model(store: EmbeddingStore, model: NeuralCollaborativeFiltering) -> None:
    """After load_from_model, embeddings come from the NCF model."""
    store.load_from_model(model)

    assert store.has_model is True

    vec = store.user_embedding(0)
    assert len(vec) == 16

    # Verify it matches the model's actual embedding.
    expected = model.get_user_embedding(0).tolist()
    assert vec == pytest.approx(expected, abs=1e-6)


def test_item_embedding_from_model(
    store: EmbeddingStore, model: NeuralCollaborativeFiltering
) -> None:
    """Item embeddings match the model's GMF item embedding table."""
    store.load_from_model(model)
    vec = store.item_embedding(5)
    expected = model.get_item_embedding(5).tolist()
    assert vec == pytest.approx(expected, abs=1e-6)


def test_out_of_range_falls_back(
    store: EmbeddingStore, model: NeuralCollaborativeFiltering
) -> None:
    """IDs outside the model's vocabulary use the fallback."""
    store.load_from_model(model)
    # model has 50 users, so user 999 is out of range.
    vec = store.user_embedding(999)
    assert len(vec) == 16


def test_batch_embeddings(store: EmbeddingStore) -> None:
    """Batch methods return one vector per ID."""
    vecs = store.user_embeddings_batch([0, 1, 2])
    assert len(vecs) == 3
    assert all(len(v) == 16 for v in vecs)


def test_clear_cache(store: EmbeddingStore) -> None:
    """clear_cache empties both user and item caches."""
    store.user_embedding(0)
    store.item_embedding(0)
    assert len(store._user_cache) == 1
    assert len(store._item_cache) == 1

    store.clear_cache()
    assert len(store._user_cache) == 0
    assert len(store._item_cache) == 0
