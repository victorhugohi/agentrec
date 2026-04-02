"""Neural Collaborative Filtering (NeuMF) model.

Implements the architecture from He et al. (2017) "Neural Collaborative Filtering":
two parallel paths — Generalized Matrix Factorization (GMF) and a Multi-Layer
Perceptron (MLP) — are fused in a NeuMF output layer for the final prediction.

Architecture diagram::

    user_id ──┬──► GMF_user_emb ──┐
              │                    ├──► element-wise multiply ──┐
    item_id ──┼──► GMF_item_emb ──┘                            │
              │                                                 ├─► concat ─► Linear(1) ─► sigmoid
              ├──► MLP_user_emb ──┐                            │
              │                    ├──► concat ──► MLP layers ──┘
              └──► MLP_item_emb ──┘
                                   [128] → [64] → [32]
                                   (Linear + ReLU + Dropout) × 3
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering model combining GMF and MLP paths.

    The GMF path captures linear user-item interactions via element-wise
    product of embeddings.  The MLP path learns non-linear interactions
    through stacked hidden layers.  Both are concatenated and passed through
    a single output neuron with sigmoid activation to predict the probability
    of interaction.

    Attributes:
        num_users: Total number of users in the dataset.
        num_items: Total number of items in the dataset.
        embedding_dim: Dimensionality of each embedding vector.
        mlp_layer_sizes: Hidden-layer widths for the MLP path.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_layers: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the NeuMF model.

        Args:
            num_users: Size of the user vocabulary.
            num_items: Size of the item vocabulary.
            embedding_dim: Shared embedding dimensionality for both GMF and
                MLP paths.  Each path has its own user and item embedding
                tables so that they can specialise independently.
            mlp_layers: Widths of the MLP hidden layers.
                Defaults to ``[128, 64, 32]``.  Each layer is a
                ``Linear → ReLU → Dropout`` block.
            dropout: Dropout probability applied after each MLP hidden layer
                and after the GMF element-wise product.  Set to 0.0 to
                disable.
        """
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.mlp_layer_sizes = list(mlp_layers)

        # ---- GMF path ----
        # Separate embeddings so GMF and MLP can learn different representations.
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        self.gmf_dropout = nn.Dropout(dropout)

        # ---- MLP path ----
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)

        # Build MLP tower: input is concatenated user + item embeddings.
        mlp_input_dim = embedding_dim * 2
        mlp_modules: list[nn.Module] = []
        for hidden_dim in mlp_layers:
            mlp_modules.append(nn.Linear(mlp_input_dim, hidden_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            mlp_input_dim = hidden_dim
        self.mlp_tower = nn.Sequential(*mlp_modules)

        # ---- NeuMF fusion layer ----
        # Concatenates GMF output (embedding_dim) with MLP output (last hidden dim).
        neumf_input_dim = embedding_dim + mlp_layers[-1]
        self.neumf_output = nn.Linear(neumf_input_dim, 1)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialize embedding and linear layers.

        Embeddings use a normal distribution (mean=0, std=0.01).
        Linear layers use Xavier uniform, biases are zeroed.
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Compute predicted interaction scores for user-item pairs.

        Runs both GMF and MLP paths in parallel, concatenates their outputs,
        and passes the result through the NeuMF fusion layer with sigmoid
        activation.

        Args:
            user_ids: Integer tensor of user indices, shape ``(batch_size,)``.
                Values must be in ``[0, num_users)``.
            item_ids: Integer tensor of item indices, shape ``(batch_size,)``.
                Values must be in ``[0, num_items)``.

        Returns:
            Predicted interaction probabilities, shape ``(batch_size, 1)``,
            with values in ``[0, 1]``.
        """
        # GMF path: element-wise product of user and item embeddings.
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_out = self.gmf_dropout(gmf_user * gmf_item)

        # MLP path: concatenate embeddings then pass through the tower.
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp_tower(mlp_input)

        # NeuMF fusion: concatenate GMF and MLP outputs → single prediction.
        neumf_input = torch.cat([gmf_out, mlp_out], dim=-1)
        logit = self.neumf_output(neumf_input)
        return torch.sigmoid(logit)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_batch(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Score user-item pairs in mini-batches without computing gradients.

        Suitable for large-scale inference where the full input may not fit
        in GPU memory at once.  The model is automatically set to eval mode
        for the duration of the call and restored afterwards.

        Args:
            user_ids: 1-D tensor of user indices, length ``N``.
            item_ids: 1-D tensor of item indices, length ``N``.
            batch_size: Number of pairs to score per forward pass.

        Returns:
            1-D tensor of predicted scores, length ``N``, values in ``[0, 1]``.
        """
        was_training = self.training
        self.eval()

        scores_list: list[torch.Tensor] = []
        n = user_ids.size(0)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_users = user_ids[start:end]
            batch_items = item_ids[start:end]
            batch_scores = self.forward(batch_users, batch_items).squeeze(-1)
            scores_list.append(batch_scores)

        if was_training:
            self.train()

        return torch.cat(scores_list, dim=0)

    @classmethod
    def load_pretrained(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> NeuralCollaborativeFiltering:
        """Load a pretrained NeuMF model from a checkpoint file.

        The checkpoint must be a dict containing:

        - ``"model_state_dict"``: the ``state_dict()`` of a
          :class:`NeuralCollaborativeFiltering` instance.
        - ``"model_config"``: a dict with ``num_users``, ``num_items``,
          ``embedding_dim``, and optionally ``mlp_layers`` / ``dropout``.

        Args:
            path: Filesystem path to the ``.pt`` checkpoint file.
            device: Device to map tensors onto (e.g. ``"cpu"`` or ``"cuda"``).

        Returns:
            A :class:`NeuralCollaborativeFiltering` instance with loaded
            weights, set to eval mode.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If required keys are missing from the checkpoint.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info("Loading pretrained NCF model from %s", path)
        checkpoint: dict = torch.load(path, map_location=device, weights_only=False)

        config: dict = checkpoint["model_config"]
        model = cls(
            num_users=config["num_users"],
            num_items=config["num_items"],
            embedding_dim=config.get("embedding_dim", 64),
            mlp_layers=config.get("mlp_layers", [128, 64, 32]),
            dropout=config.get("dropout", 0.2),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info(
            "Loaded NCF model: %d users, %d items, dim=%d",
            model.num_users,
            model.num_items,
            model.embedding_dim,
        )
        return model

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights and config to a checkpoint file.

        Args:
            path: Destination filepath (typically ending in ``.pt``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_config": {
                "num_users": self.num_users,
                "num_items": self.num_items,
                "embedding_dim": self.embedding_dim,
                "mlp_layers": self.mlp_layer_sizes,
            },
            "model_state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info("Saved NCF checkpoint to %s", path)

    # ------------------------------------------------------------------
    # Embedding access (used by the shared embedding store)
    # ------------------------------------------------------------------

    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """Retrieve the GMF user embedding for a single user.

        Args:
            user_id: User index in ``[0, num_users)``.

        Returns:
            1-D tensor of shape ``(embedding_dim,)``.
        """
        idx = torch.tensor([user_id], dtype=torch.long, device=self._device())
        return self.gmf_user_embedding(idx).squeeze(0)

    def get_item_embedding(self, item_id: int) -> torch.Tensor:
        """Retrieve the GMF item embedding for a single item.

        Args:
            item_id: Item index in ``[0, num_items)``.

        Returns:
            1-D tensor of shape ``(embedding_dim,)``.
        """
        idx = torch.tensor([item_id], dtype=torch.long, device=self._device())
        return self.gmf_item_embedding(idx).squeeze(0)

    def _device(self) -> torch.device:
        """Return the device of the model parameters."""
        return next(self.parameters()).device
