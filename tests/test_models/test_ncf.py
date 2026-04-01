"""Tests for the Neural Collaborative Filtering model."""

import tempfile
from pathlib import Path

import torch
import pytest

from src.models.ncf import NeuralCollaborativeFiltering


@pytest.fixture
def model() -> NeuralCollaborativeFiltering:
    """Return a small NCF model for testing."""
    return NeuralCollaborativeFiltering(
        num_users=100, num_items=50, embedding_dim=32, mlp_layers=[64, 32, 16]
    )


def test_forward_shape(model: NeuralCollaborativeFiltering) -> None:
    """Forward pass returns (batch_size, 1) output."""
    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([10, 20, 30])

    output = model(user_ids, item_ids)

    assert output.shape == (3, 1)


def test_output_range(model: NeuralCollaborativeFiltering) -> None:
    """Output values are between 0 and 1 (sigmoid)."""
    user_ids = torch.tensor([0, 1])
    item_ids = torch.tensor([0, 1])

    output = model(user_ids, item_ids)

    assert (output >= 0).all()
    assert (output <= 1).all()


def test_predict_batch_matches_forward(model: NeuralCollaborativeFiltering) -> None:
    """predict_batch produces the same scores as forward in eval mode."""
    user_ids = torch.tensor([0, 5, 10, 15, 20])
    item_ids = torch.tensor([1, 2, 3, 4, 5])

    model.eval()
    with torch.no_grad():
        expected = model(user_ids, item_ids).squeeze(-1)

    actual = model.predict_batch(user_ids, item_ids, batch_size=2)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_predict_batch_single_item(model: NeuralCollaborativeFiltering) -> None:
    """predict_batch works with a single user-item pair."""
    scores = model.predict_batch(
        torch.tensor([0]), torch.tensor([0]), batch_size=1
    )
    assert scores.shape == (1,)
    assert 0.0 <= scores.item() <= 1.0


def test_get_user_embedding(model: NeuralCollaborativeFiltering) -> None:
    """get_user_embedding returns a vector of the correct dimension."""
    vec = model.get_user_embedding(0)
    assert vec.shape == (32,)


def test_get_item_embedding(model: NeuralCollaborativeFiltering) -> None:
    """get_item_embedding returns a vector of the correct dimension."""
    vec = model.get_item_embedding(0)
    assert vec.shape == (32,)


def test_save_and_load_checkpoint(model: NeuralCollaborativeFiltering) -> None:
    """Round-trip save → load preserves model weights and config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pt"
        model.save_checkpoint(path)

        loaded = NeuralCollaborativeFiltering.load_pretrained(path)

    assert loaded.num_users == model.num_users
    assert loaded.num_items == model.num_items
    assert loaded.embedding_dim == model.embedding_dim

    # Weights should be identical.
    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([3, 4, 5])
    model.eval()
    loaded.eval()
    with torch.no_grad():
        original_out = model(user_ids, item_ids)
        loaded_out = loaded(user_ids, item_ids)
    assert torch.allclose(original_out, loaded_out)


def test_load_pretrained_missing_file() -> None:
    """load_pretrained raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        NeuralCollaborativeFiltering.load_pretrained("/nonexistent/model.pt")


def test_custom_mlp_layers() -> None:
    """Model respects custom MLP layer sizes."""
    model = NeuralCollaborativeFiltering(
        num_users=10, num_items=10, embedding_dim=16, mlp_layers=[64, 16]
    )
    assert model.mlp_layer_sizes == [64, 16]

    output = model(torch.tensor([0]), torch.tensor([0]))
    assert output.shape == (1, 1)


def test_dropout_zero_deterministic() -> None:
    """With dropout=0 the model is deterministic even in train mode."""
    model = NeuralCollaborativeFiltering(
        num_users=10, num_items=10, embedding_dim=8, dropout=0.0
    )
    model.train()
    u = torch.tensor([0, 1])
    i = torch.tensor([0, 1])
    out1 = model(u, i)
    out2 = model(u, i)
    assert torch.allclose(out1, out2)
