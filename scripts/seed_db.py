#!/usr/bin/env python3
"""Seed the PostgreSQL database from processed Parquet files.

Reads the processed MovieLens splits, creates tables if they don't exist,
truncates existing data, and bulk-inserts all users, movies, and ratings
using asyncpg COPY for speed.

Usage::

    python scripts/seed_db.py
    python scripts/seed_db.py --database-url postgresql://user:pass@host/db
    python scripts/seed_db.py --variant 25m --data-dir /data
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import (
    bulk_insert_movies,
    bulk_insert_ratings,
    bulk_insert_users,
    close_db,
    create_tables,
    init_db,
    get_pool,
)
from src.data.loader import MovieLensLoader

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


async def seed(args: argparse.Namespace) -> None:
    """Run the full seed pipeline."""
    # Connect
    await init_db(args.database_url, min_size=2, max_size=5)
    pool = get_pool()
    assert pool is not None

    # Load processed data
    loader = MovieLensLoader(variant=args.variant, data_dir=args.data_dir)
    splits = loader.load_processed()
    if splits is None:
        logger.error(
            "No processed data found in %s.  Run scripts/train_model.py first "
            "or call loader.preprocess() to generate the Parquet files.",
            args.data_dir,
        )
        await close_db()
        sys.exit(1)

    logger.info(
        "Loaded splits: %d users, %d movies, %d ratings",
        splits.num_users,
        splits.num_movies,
        len(splits.train) + len(splits.val) + len(splits.test),
    )

    # Create tables
    await create_tables(pool)

    # Truncate in dependency order
    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE ratings, recommendations_log, movies, users CASCADE"
        )
    logger.info("Truncated existing data")

    # --- Insert users ---
    all_ratings = pd.concat(
        [splits.train, splits.val, splits.test], ignore_index=True,
    )
    user_ids = sorted(all_ratings["userId"].unique().tolist())
    await bulk_insert_users(pool, user_ids)

    # --- Insert movies ---
    movies: list[dict] = []
    for _, row in splits.movies.iterrows():
        title = str(row.get("title", ""))
        genres_raw = row.get("genres", "")
        if isinstance(genres_raw, str):
            genres = [g for g in genres_raw.split("|") if g and g != "(no genres listed)"]
        else:
            genres = genres_raw if isinstance(genres_raw, list) else []

        year_match = _YEAR_RE.search(title)
        year = int(year_match.group(1)) if year_match else 0

        movies.append({
            "movie_id": int(row["movieId"]),
            "title": title,
            "genres": genres,
            "year": year,
        })
    await bulk_insert_movies(pool, movies)

    # --- Insert ratings ---
    rating_tuples: list[tuple[int, int, float, int]] = [
        (int(row["userId"]), int(row["movieId"]), float(row["rating"]), int(row["timestamp"]))
        for _, row in all_ratings.iterrows()
    ]
    await bulk_insert_ratings(pool, rating_tuples)

    # Summary
    async with pool.acquire() as conn:
        u = await conn.fetchval("SELECT count(*) FROM users")
        m = await conn.fetchval("SELECT count(*) FROM movies")
        r = await conn.fetchval("SELECT count(*) FROM ratings")
    logger.info("Seed complete: %d users, %d movies, %d ratings", u, m, r)

    await close_db()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Seed PostgreSQL from Parquet files")
    p.add_argument(
        "--database-url",
        default="postgresql://postgres:postgres@localhost:5432/agentrec",
        help="PostgreSQL DSN",
    )
    p.add_argument("--variant", default="small", choices=["small", "25m"])
    p.add_argument("--data-dir", default="data")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(seed(parse_args()))
