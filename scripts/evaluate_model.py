#!/usr/bin/env python3
"""Evaluate a trained NCF model on the test set.

Computes NDCG@10, Hit Rate@10, MRR, and Coverage.

Usage::

    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --checkpoint models/checkpoints/ncf_best.pt --k 20
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import MovieLensLoader
from src.models.ncf import NeuralCollaborativeFiltering

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    model: NeuralCollaborativeFiltering,
    test_df: pd.DataFrame,
    n_items: int,
    k: int = 10,
    relevance_threshold: float = 0.7,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics on *test_df*.

    Args:
        model: Trained NCF model in eval mode.
        test_df: Test split DataFrame with userId, movieId, rating columns.
        n_items: Total number of items in the dataset.
        k: Cut-off for top-k metrics.
        relevance_threshold: Minimum normalised rating to count as relevant.
        device: Torch device.

    Returns:
        Dict with keys ``ndcg``, ``hit_rate``, ``mrr``, ``coverage``.
    """
    if device is None:
        device = next(model.parameters()).device

    ndcg_sum = 0.0
    hit_count = 0
    mrr_sum = 0.0
    n_users = 0
    recommended_items: set[int] = set()

    for uid, group in test_df.groupby("userId"):
        items = torch.tensor(group["movieId"].values, dtype=torch.long, device=device)
        ratings = group["rating"].values
        users = torch.full_like(items, int(uid))

        scores = model.predict_batch(users, items).cpu().numpy()
        ranked_idx = scores.argsort()[::-1]
        top_k_idx = ranked_idx[:k]

        # Track recommended items for coverage
        for idx in top_k_idx:
            recommended_items.add(int(group["movieId"].values[idx]))

        # NDCG@k
        dcg = sum(
            ratings[idx] / math.log2(rank + 2)
            for rank, idx in enumerate(top_k_idx)
        )
        ideal_order = ratings.argsort()[::-1][:k]
        idcg = sum(
            ratings[idx] / math.log2(rank + 2)
            for rank, idx in enumerate(ideal_order)
        )
        if idcg > 0:
            ndcg_sum += dcg / idcg

        # Hit Rate@k
        if any(ratings[idx] >= relevance_threshold for idx in top_k_idx):
            hit_count += 1

        # MRR (Mean Reciprocal Rank) — rank of the first relevant item
        for rank, idx in enumerate(ranked_idx):
            if ratings[idx] >= relevance_threshold:
                mrr_sum += 1.0 / (rank + 1)
                break

        n_users += 1

    n_users = max(n_users, 1)
    coverage = len(recommended_items) / max(n_items, 1)

    return {
        "ndcg": ndcg_sum / n_users,
        "hit_rate": hit_count / n_users,
        "mrr": mrr_sum / n_users,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> None:
    """Load checkpoint and evaluate on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    loader = MovieLensLoader(variant=args.variant, data_dir=args.data_dir)
    splits = loader.load_processed()
    if splits is None:
        logger.error(
            "No processed data found in %s. Run train_model.py first.", args.data_dir
        )
        sys.exit(1)

    logger.info(
        "Test set: %d ratings, %d users, %d items",
        len(splits.test), splits.num_users, splits.num_movies,
    )

    # Load model
    checkpoint_path = Path(args.checkpoint)
    model = NeuralCollaborativeFiltering.load_pretrained(checkpoint_path, device=device)
    logger.info("Loaded model from %s", checkpoint_path)

    # Evaluate
    metrics = compute_metrics(
        model=model,
        test_df=splits.test,
        n_items=splits.num_movies,
        k=args.k,
        relevance_threshold=args.threshold,
        device=device,
    )

    logger.info("=" * 50)
    logger.info("Evaluation Results (k=%d)", args.k)
    logger.info("=" * 50)
    logger.info("  NDCG@%d:     %.4f", args.k, metrics["ndcg"])
    logger.info("  HR@%d:       %.4f", args.k, metrics["hit_rate"])
    logger.info("  MRR:          %.4f", metrics["mrr"])
    logger.info("  Coverage:     %.4f (%.1f%%)", metrics["coverage"], metrics["coverage"] * 100)
    logger.info("=" * 50)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate NCF on MovieLens test set")
    p.add_argument(
        "--checkpoint", default="models/checkpoints/ncf_best.pt",
        help="Path to model checkpoint",
    )
    p.add_argument("--variant", default="small", choices=["small", "25m"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--k", type=int, default=10, help="Top-k cut-off")
    p.add_argument("--threshold", type=float, default=0.7,
                    help="Relevance threshold for HR and MRR")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    evaluate(parse_args())
