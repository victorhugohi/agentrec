#!/usr/bin/env python3
"""Train the Neural Collaborative Filtering model on MovieLens data.

Usage::

    python scripts/train_model.py --variant small --epochs 30 --batch-size 256
    python scripts/train_model.py --embedding-dim 32 --lr 0.002
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running as ``python scripts/train_model.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import MovieLensLoader
from src.data.movielens import MovieLensDataset
from src.models.ncf import NeuralCollaborativeFiltering

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("models/checkpoints")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _ndcg_at_k(
    model: NeuralCollaborativeFiltering,
    df: pd.DataFrame,
    k: int = 10,
    device: torch.device | None = None,
) -> float:
    """Compute NDCG@k averaged over all users in *df*."""
    if device is None:
        device = next(model.parameters()).device

    ndcg_sum = 0.0
    n_users = 0

    for uid, group in df.groupby("userId"):
        items = torch.tensor(group["movieId"].values, dtype=torch.long, device=device)
        ratings = group["rating"].values
        users = torch.full_like(items, int(uid))

        scores = model.predict_batch(users, items).cpu().numpy()

        # Rank items by predicted score (descending)
        top_k_idx = scores.argsort()[::-1][:k]
        dcg = sum(
            ratings[idx] / math.log2(rank + 2)
            for rank, idx in enumerate(top_k_idx)
        )

        # Ideal DCG: sort by true rating
        ideal_order = ratings.argsort()[::-1][:k]
        idcg = sum(
            ratings[idx] / math.log2(rank + 2)
            for rank, idx in enumerate(ideal_order)
        )

        if idcg > 0:
            ndcg_sum += dcg / idcg
            n_users += 1

    return ndcg_sum / max(n_users, 1)


def _hit_rate_at_k(
    model: NeuralCollaborativeFiltering,
    df: pd.DataFrame,
    k: int = 10,
    threshold: float = 0.7,
    device: torch.device | None = None,
) -> float:
    """Compute Hit Rate@k: fraction of users whose top-k contains a relevant item."""
    if device is None:
        device = next(model.parameters()).device

    hits = 0
    n_users = 0

    for uid, group in df.groupby("userId"):
        items = torch.tensor(group["movieId"].values, dtype=torch.long, device=device)
        ratings = group["rating"].values
        users = torch.full_like(items, int(uid))

        scores = model.predict_batch(users, items).cpu().numpy()
        top_k_idx = scores.argsort()[::-1][:k]

        if any(ratings[idx] >= threshold for idx in top_k_idx):
            hits += 1
        n_users += 1

    return hits / max(n_users, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run the full training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- Load data --------------------------------------------------------
    loader = MovieLensLoader(variant=args.variant, data_dir=args.data_dir)
    splits = loader.load_processed()
    if splits is None:
        logger.info("No processed data found — downloading and preprocessing …")
        loader.download()
        splits = loader.preprocess(min_ratings=args.min_ratings)

    n_users = splits.num_users
    n_items = splits.num_movies
    logger.info("Dataset: %d users, %d items", n_users, n_items)
    logger.info(
        "Splits: train=%d, val=%d, test=%d",
        len(splits.train), len(splits.val), len(splits.test),
    )

    train_ds = MovieLensDataset(splits.train, n_users, n_items)
    val_ds = MovieLensDataset(splits.val, n_users, n_items)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ---- Build model ------------------------------------------------------
    mlp_layers = [int(x) for x in args.mlp_layers.split(",")]

    model = NeuralCollaborativeFiltering(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=args.embedding_dim,
        mlp_layers=mlp_layers,
        dropout=args.dropout,
    ).to(device)

    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # The NCF forward() applies sigmoid, so use BCELoss (not BCEWithLogitsLoss)
    # to match the user's intent of treating this as implicit-feedback binary
    # classification.  If the user truly needs logits, we'd need to modify
    # the model, but BCELoss on sigmoid output is mathematically equivalent.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- Training loop ----------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / "ncf_best.pt"

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        n_batches = 0

        for users, items, ratings in train_loader:
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            preds = model(users, items).squeeze(-1)
            loss = criterion(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for users, items, ratings in val_loader:
                users = users.to(device)
                items = items.to(device)
                ratings = ratings.to(device)

                preds = model(users, items).squeeze(-1)
                loss = criterion(preds, ratings)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)

        # -- Metrics (compute every few epochs to save time) --
        if epoch % args.eval_every == 0 or epoch == 1:
            ndcg = _ndcg_at_k(model, splits.val, k=10, device=device)
            hr = _hit_rate_at_k(model, splits.val, k=10, device=device)
            logger.info(
                "Epoch %03d | train_loss=%.4f | val_loss=%.4f | "
                "NDCG@10=%.4f | HR@10=%.4f",
                epoch, avg_train_loss, avg_val_loss, ndcg, hr,
            )
        else:
            logger.info(
                "Epoch %03d | train_loss=%.4f | val_loss=%.4f",
                epoch, avg_train_loss, avg_val_loss,
            )

        # -- Early stopping --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_checkpoint(best_path)
            logger.info("  ↳ Saved best model (val_loss=%.4f)", avg_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch, args.patience,
                )
                break

    logger.info("Training complete. Best checkpoint: %s", best_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Train NCF on MovieLens data")

    # Data
    p.add_argument("--variant", default="small", choices=["small", "25m"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--min-ratings", type=int, default=20)

    # Model
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--mlp-layers", default="128,64,32",
                    help="Comma-separated MLP hidden layer sizes")
    p.add_argument("--dropout", type=float, default=0.2)

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=5,
                    help="Compute ranking metrics every N epochs")

    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    train(parse_args())
