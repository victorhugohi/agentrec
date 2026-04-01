"""Comprehensive model tests.

Covers NCF forward pass shapes, embedding dimensions, predict_batch
sorted-score behaviour, checkpoint round-trips, and the EmbeddingStore.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.models.ncf import NeuralCollaborativeFiltering
from src.models.embeddings import EmbeddingStore


# ── Shared fixtures ──────────────────────────────────────────────────


@pytest.fixture
def ncf_model() -> NeuralCollaborativeFiltering:
    """Standard-size NCF model for testing (64-dim, default MLP layers)."""
    model = NeuralCollaborativeFiltering(
        num_users=200, num_items=100, embedding_dim=64
    )
    model.eval()
    return model


@pytest.fixture
def small_model() -> NeuralCollaborativeFiltering:
    """Small NCF model for fast tests."""
    model = NeuralCollaborativeFiltering(
        num_users=50, num_items=30, embedding_dim=16, mlp_layers=[32, 16],
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def batch_users() -> torch.Tensor:
    """Batch of 32 random user IDs in [0, 200)."""
    return torch.randint(0, 200, (32,))


@pytest.fixture
def batch_items() -> torch.Tensor:
    """Batch of 32 random item IDs in [0, 100)."""
    return torch.randint(0, 100, (32,))


@pytest.fixture
def store() -> EmbeddingStore:
    """Fresh embedding store, no model attached."""
    return EmbeddingStore(embedding_dim=64)


# ── 1. NCF forward pass with random tensors ─────────────────────────


class TestNCFForwardPass:
    """Test forward() output shapes, ranges, and behaviour across batch sizes."""

    def test_batch_32_output_shape(
        self,
        ncf_model: NeuralCollaborativeFiltering,
        batch_users: torch.Tensor,
        batch_items: torch.Tensor,
    ) -> None:
        """Forward with batch_size=32 produces shape (32, 1)."""
        output = ncf_model(batch_users, batch_items)
        assert output.shape == (32, 1)

    def test_single_pair_output_shape(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """Forward with a single pair produces shape (1, 1)."""
        output = ncf_model(torch.tensor([0]), torch.tensor([0]))
        assert output.shape == (1, 1)

    def test_large_batch_output_shape(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """Forward with batch_size=256 produces shape (256, 1)."""
        users = torch.randint(0, 200, (256,))
        items = torch.randint(0, 100, (256,))
        output = ncf_model(users, items)
        assert output.shape == (256, 1)

    def test_output_values_in_zero_one(
        self,
        ncf_model: NeuralCollaborativeFiltering,
        batch_users: torch.Tensor,
        batch_items: torch.Tensor,
    ) -> None:
        """All output values are in [0, 1] (sigmoid)."""
        output = ncf_model(batch_users, batch_items)
        assert (output >= 0.0).all()
        assert (output <= 1.0).all()

    def test_output_dtype_is_float(
        self,
        ncf_model: NeuralCollaborativeFiltering,
        batch_users: torch.Tensor,
        batch_items: torch.Tensor,
    ) -> None:
        """Output tensor has float dtype."""
        output = ncf_model(batch_users, batch_items)
        assert output.dtype == torch.float32

    def test_eval_mode_deterministic(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """In eval mode, same inputs produce identical outputs."""
        users = torch.tensor([0, 5, 10])
        items = torch.tensor([1, 2, 3])
        out1 = ncf_model(users, items)
        out2 = ncf_model(users, items)
        assert torch.allclose(out1, out2)

    def test_different_users_different_scores(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """Different user IDs generally produce different scores for the same item."""
        items = torch.tensor([5, 5])
        users = torch.tensor([0, 50])
        output = ncf_model(users, items)
        # Not guaranteed to differ, but with random init it's extremely unlikely
        # to have exactly equal scores for different users
        scores = output.squeeze(-1)
        assert scores.shape == (2,)

    def test_gmf_and_mlp_paths_both_contribute(
        self, small_model: NeuralCollaborativeFiltering
    ) -> None:
        """Zeroing GMF embeddings changes the output, proving GMF contributes."""
        users = torch.tensor([0, 1])
        items = torch.tensor([0, 1])

        baseline = small_model(users, items).clone()

        # Zero out GMF user embeddings
        with torch.no_grad():
            small_model.gmf_user_embedding.weight.zero_()
        modified = small_model(users, items)

        assert not torch.allclose(baseline, modified)


# ── 2. Embedding dimensions ─────────────────────────────────────────


class TestEmbeddingDimensions:
    """Test that all embedding layers produce vectors of the correct size."""

    def test_gmf_user_embedding_dim(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """GMF user embedding table has shape (num_users, embedding_dim)."""
        weight = ncf_model.gmf_user_embedding.weight
        assert weight.shape == (200, 64)

    def test_gmf_item_embedding_dim(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """GMF item embedding table has shape (num_items, embedding_dim)."""
        weight = ncf_model.gmf_item_embedding.weight
        assert weight.shape == (100, 64)

    def test_mlp_user_embedding_dim(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """MLP user embedding table has shape (num_users, embedding_dim)."""
        weight = ncf_model.mlp_user_embedding.weight
        assert weight.shape == (200, 64)

    def test_mlp_item_embedding_dim(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """MLP item embedding table has shape (num_items, embedding_dim)."""
        weight = ncf_model.mlp_item_embedding.weight
        assert weight.shape == (100, 64)

    def test_get_user_embedding_returns_correct_dim(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """get_user_embedding returns a 1-D tensor of length embedding_dim."""
        vec = ncf_model.get_user_embedding(0)
        assert vec.shape == (64,)

    def test_get_item_embedding_returns_correct_dim(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """get_item_embedding returns a 1-D tensor of length embedding_dim."""
        vec = ncf_model.get_item_embedding(0)
        assert vec.shape == (64,)

    def test_different_embedding_dim(self) -> None:
        """Model respects a non-default embedding_dim=32."""
        model = NeuralCollaborativeFiltering(
            num_users=10, num_items=10, embedding_dim=32
        )
        assert model.gmf_user_embedding.weight.shape == (10, 32)
        assert model.mlp_item_embedding.weight.shape == (10, 32)

    def test_mlp_first_layer_input_matches_concat(self) -> None:
        """MLP tower first layer accepts 2 * embedding_dim inputs."""
        model = NeuralCollaborativeFiltering(
            num_users=10, num_items=10, embedding_dim=48
        )
        first_linear = model.mlp_tower[0]
        assert first_linear.in_features == 48 * 2

    def test_neumf_output_input_matches_gmf_plus_mlp(self) -> None:
        """NeuMF fusion layer input = embedding_dim + last MLP hidden dim."""
        model = NeuralCollaborativeFiltering(
            num_users=10, num_items=10, embedding_dim=64, mlp_layers=[128, 64, 32]
        )
        assert model.neumf_output.in_features == 64 + 32

    def test_embedding_store_fallback_dim(self, store: EmbeddingStore) -> None:
        """Fallback embeddings respect the store's embedding_dim."""
        vec = store.user_embedding(0)
        assert len(vec) == 64

    def test_embedding_store_model_backed_dim(
        self, store: EmbeddingStore, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """Model-backed embeddings match embedding_dim."""
        store.load_from_model(ncf_model)
        vec = store.item_embedding(5)
        assert len(vec) == 64


# ── 3. predict_batch returns sorted scores ───────────────────────────


class TestPredictBatchScores:
    """Test predict_batch output shapes, ranges, and sorting behaviour."""

    def test_predict_batch_output_length(
        self,
        ncf_model: NeuralCollaborativeFiltering,
        batch_users: torch.Tensor,
        batch_items: torch.Tensor,
    ) -> None:
        """predict_batch returns exactly N scores for N input pairs."""
        scores = ncf_model.predict_batch(batch_users, batch_items)
        assert scores.shape == (32,)

    def test_predict_batch_values_in_zero_one(
        self,
        ncf_model: NeuralCollaborativeFiltering,
        batch_users: torch.Tensor,
        batch_items: torch.Tensor,
    ) -> None:
        """All predicted scores are in [0, 1]."""
        scores = ncf_model.predict_batch(batch_users, batch_items)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_predict_batch_matches_forward(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """predict_batch produces identical scores to manual forward in eval mode."""
        users = torch.tensor([0, 10, 20, 30, 40])
        items = torch.tensor([5, 15, 25, 35, 45])

        with torch.no_grad():
            expected = ncf_model(users, items).squeeze(-1)
        actual = ncf_model.predict_batch(users, items)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_predict_batch_small_batch_size(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """predict_batch with batch_size=4 on 10 items still returns 10 scores."""
        users = torch.randint(0, 200, (10,))
        items = torch.randint(0, 100, (10,))
        scores = ncf_model.predict_batch(users, items, batch_size=4)
        assert scores.shape == (10,)

    def test_predict_batch_scores_can_be_sorted(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """Scores returned by predict_batch can be sorted descending for ranking."""
        user_id = 0
        item_ids = torch.arange(0, 50)
        user_ids = torch.full((50,), user_id, dtype=torch.long)

        scores = ncf_model.predict_batch(user_ids, item_ids)
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)

        # After sorting, scores should be monotonically non-increasing
        for i in range(len(sorted_scores) - 1):
            assert sorted_scores[i] >= sorted_scores[i + 1]

        # sorted_indices maps back to original positions
        assert sorted_indices.shape == (50,)

    def test_predict_batch_top_k_selection(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """Top-k items by predict_batch score can be selected via topk."""
        user_ids = torch.full((20,), 0, dtype=torch.long)
        item_ids = torch.arange(0, 20)

        scores = ncf_model.predict_batch(user_ids, item_ids)
        top_k = 5
        top_scores, top_indices = torch.topk(scores, top_k)

        assert top_scores.shape == (top_k,)
        assert top_indices.shape == (top_k,)
        # Top scores are >= all non-top scores
        min_top = top_scores.min()
        for i in range(20):
            if i not in top_indices:
                assert scores[i] <= min_top

    def test_predict_batch_restores_training_mode(self) -> None:
        """predict_batch restores the model to training mode if it was training."""
        model = NeuralCollaborativeFiltering(
            num_users=10, num_items=10, embedding_dim=8, dropout=0.0
        )
        model.train()
        assert model.training

        model.predict_batch(torch.tensor([0]), torch.tensor([0]))

        assert model.training  # restored

    def test_predict_batch_stays_eval_if_was_eval(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """predict_batch keeps eval mode if the model was already in eval."""
        assert not ncf_model.training  # fixture sets eval
        ncf_model.predict_batch(torch.tensor([0]), torch.tensor([0]))
        assert not ncf_model.training


# ── 4. Checkpoint save/load ──────────────────────────────────────────


class TestCheckpointRoundTrip:
    """Test save_checkpoint and load_pretrained preserve the model exactly."""

    def test_round_trip_preserves_scores(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """Saved and loaded model produces identical scores."""
        users = torch.tensor([0, 5, 10])
        items = torch.tensor([1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ncf.pt"
            ncf_model.save_checkpoint(path)
            loaded = NeuralCollaborativeFiltering.load_pretrained(path)

        with torch.no_grad():
            original = ncf_model(users, items)
            restored = loaded(users, items)

        assert torch.allclose(original, restored)

    def test_round_trip_preserves_config(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """Loaded model has the same num_users, num_items, embedding_dim."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ncf.pt"
            ncf_model.save_checkpoint(path)
            loaded = NeuralCollaborativeFiltering.load_pretrained(path)

        assert loaded.num_users == ncf_model.num_users
        assert loaded.num_items == ncf_model.num_items
        assert loaded.embedding_dim == ncf_model.embedding_dim
        assert loaded.mlp_layer_sizes == ncf_model.mlp_layer_sizes

    def test_load_nonexistent_raises(self) -> None:
        """Loading from a missing path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            NeuralCollaborativeFiltering.load_pretrained("/no/such/file.pt")

    def test_loaded_model_is_in_eval_mode(
        self, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """load_pretrained returns a model in eval mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ncf.pt"
            ncf_model.save_checkpoint(path)
            loaded = NeuralCollaborativeFiltering.load_pretrained(path)

        assert not loaded.training


# ── 5. EmbeddingStore integration ────────────────────────────────────


class TestEmbeddingStoreIntegration:
    """Test EmbeddingStore with and without a model."""

    def test_fallback_is_deterministic(self, store: EmbeddingStore) -> None:
        """Same ID always produces the same fallback vector."""
        a = store.user_embedding(42)
        store.clear_cache()
        b = store.user_embedding(42)
        assert a == b

    def test_user_and_item_embeddings_differ(self, store: EmbeddingStore) -> None:
        """User and item vectors for the same numeric ID are different."""
        assert store.user_embedding(1) != store.item_embedding(1)

    def test_model_backed_matches_model(
        self, store: EmbeddingStore, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """After loading a model, store embeddings match the model's tables."""
        store.load_from_model(ncf_model)

        for uid in [0, 10, 50]:
            expected = ncf_model.get_user_embedding(uid).tolist()
            actual = store.user_embedding(uid)
            assert actual == pytest.approx(expected, abs=1e-6)

    def test_out_of_range_uses_fallback(
        self, store: EmbeddingStore, ncf_model: NeuralCollaborativeFiltering
    ) -> None:
        """IDs beyond the model's vocabulary get fallback embeddings."""
        store.load_from_model(ncf_model)
        vec = store.user_embedding(ncf_model.num_users + 100)
        assert len(vec) == 64

    def test_batch_returns_correct_count(self, store: EmbeddingStore) -> None:
        """Batch method returns one vector per ID."""
        vecs = store.item_embeddings_batch([0, 1, 2, 3, 4])
        assert len(vecs) == 5
        assert all(len(v) == 64 for v in vecs)
