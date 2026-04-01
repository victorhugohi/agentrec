"""API route definitions for AgentRec."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import asyncpg
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from src.agents.orchestrator import OrchestratorAgent
from src.api.dependencies import get_db_pool, get_model, get_orchestrator, get_splits
from src.api.middleware import metrics_collector, request_id_var
from src.data.loader import DataSplits
from src.models.ncf import NeuralCollaborativeFiltering
from src.models.schemas import (
    AgentHealth,
    ErrorResponse,
    HealthResponse,
    MovieItem,
    MovieSearchResponse,
    MovieSearchResult,
    Rating,
    RecommendationRequest,
    RecommendationResponse,
    UserHistoryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recommendations"])

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _error_response(status: int, error: str, message: str) -> HTTPException:
    """Build an HTTPException with a structured JSON body."""
    rid = request_id_var.get("")
    return HTTPException(
        status_code=status,
        detail=ErrorResponse(error=error, message=message, request_id=rid).model_dump(),
    )


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def recommend(
    request: RecommendationRequest,
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
    model: NeuralCollaborativeFiltering | None = Depends(get_model),
) -> RecommendationResponse:
    """Generate personalised recommendations for a user.

    Runs the full orchestrator pipeline (profile → content → scoring)
    and returns top-N recommendations with scores.
    """
    start = time.perf_counter()

    result = await orchestrator.run({
        "user_id": request.user_id,
        "top_k": request.top_k,
    })

    duration = time.perf_counter() - start
    metrics_collector.record_model_inference(duration)

    recommendations = [MovieItem(**item) for item in result["recommendations"]]

    return RecommendationResponse(
        user_id=result["user_id"],
        recommendations=recommendations,
        model=result["metadata"]["model"],
    )


# ---------------------------------------------------------------------------
# GET /recommend/{user_id}
# ---------------------------------------------------------------------------


@router.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    responses={503: {"model": ErrorResponse}},
)
async def recommend_simple(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=100),
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
) -> RecommendationResponse:
    """Simple recommendation endpoint with sensible defaults."""
    start = time.perf_counter()

    result = await orchestrator.run({
        "user_id": user_id,
        "top_k": top_k,
    })

    duration = time.perf_counter() - start
    metrics_collector.record_model_inference(duration)

    recommendations = [MovieItem(**item) for item in result["recommendations"]]

    return RecommendationResponse(
        user_id=result["user_id"],
        recommendations=recommendations,
        model=result["metadata"]["model"],
    )


# ---------------------------------------------------------------------------
# GET /user/{user_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/user/{user_id}/history",
    response_model=UserHistoryResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def user_history(
    user_id: int,
    pool: asyncpg.Pool | None = Depends(get_db_pool),
    splits: DataSplits | None = Depends(get_splits),
) -> UserHistoryResponse:
    """Return the rating history for a user."""
    if pool is not None:
        return await _user_history_from_db(pool, user_id)
    if splits is not None:
        return _user_history_from_splits(splits, user_id)
    raise _error_response(503, "data_unavailable", "No data loaded.")


async def _user_history_from_db(pool: asyncpg.Pool, user_id: int) -> UserHistoryResponse:
    """Fetch user history from PostgreSQL."""
    from src.data.database import fetch_user_ratings

    rows = await fetch_user_ratings(pool, user_id)
    if not rows:
        raise _error_response(404, "user_not_found", f"User {user_id} not found.")

    ratings = [
        Rating(
            item_id=r["movie_id"],
            rating=r["rating"],
            title=r["title"],
            genres=r["genres"],
        )
        for r in rows
    ]
    avg = sum(r.rating for r in ratings) / len(ratings)
    return UserHistoryResponse(
        user_id=user_id,
        ratings=ratings,
        rating_count=len(ratings),
        avg_rating=round(avg, 4),
    )


def _user_history_from_splits(splits: DataSplits, user_id: int) -> UserHistoryResponse:
    """Fetch user history from in-memory DataSplits."""
    all_ratings = pd.concat(
        [splits.train, splits.val, splits.test], ignore_index=True,
    )
    user_rows = all_ratings[all_ratings["userId"] == user_id]

    if user_rows.empty:
        raise _error_response(404, "user_not_found", f"User {user_id} not found.")

    merged = user_rows.merge(splits.movies, on="movieId", how="left")

    ratings: list[Rating] = []
    for _, row in merged.iterrows():
        genres_raw = row.get("genres", "")
        if isinstance(genres_raw, str):
            genres = [g for g in genres_raw.split("|") if g and g != "(no genres listed)"]
        else:
            genres = genres_raw if isinstance(genres_raw, list) else []

        ratings.append(Rating(
            item_id=int(row["movieId"]),
            rating=float(row["rating"]),
            title=str(row.get("title", "")),
            genres=genres,
        ))

    avg = sum(r.rating for r in ratings) / len(ratings) if ratings else 0.0

    return UserHistoryResponse(
        user_id=user_id,
        ratings=ratings,
        rating_count=len(ratings),
        avg_rating=round(avg, 4),
    )


# ---------------------------------------------------------------------------
# GET /movies/search
# ---------------------------------------------------------------------------


@router.get(
    "/movies/search",
    response_model=MovieSearchResponse,
    responses={503: {"model": ErrorResponse}},
)
async def movie_search(
    q: str = Query(..., min_length=1, description="Search query for movie title"),
    limit: int = Query(default=20, ge=1, le=100),
    pool: asyncpg.Pool | None = Depends(get_db_pool),
    splits: DataSplits | None = Depends(get_splits),
) -> MovieSearchResponse:
    """Search movies by title (case-insensitive substring match)."""
    if pool is not None:
        return await _movie_search_from_db(pool, q, limit)
    if splits is not None:
        return _movie_search_from_splits(splits, q, limit)
    raise _error_response(503, "data_unavailable", "No data loaded.")


async def _movie_search_from_db(
    pool: asyncpg.Pool, q: str, limit: int,
) -> MovieSearchResponse:
    """Search movies via PostgreSQL ILIKE."""
    from src.data.database import search_movies

    rows = await search_movies(pool, q, limit)
    results = [
        MovieSearchResult(
            movie_id=r["movie_id"],
            title=r["title"],
            genres=r["genres"],
            year=r["year"],
        )
        for r in rows
    ]
    return MovieSearchResponse(query=q, results=results, total=len(results))


def _movie_search_from_splits(
    splits: DataSplits, q: str, limit: int,
) -> MovieSearchResponse:
    """Search movies from in-memory DataSplits."""
    query_lower = q.lower()
    matches: list[MovieSearchResult] = []

    for _, row in splits.movies.iterrows():
        title = str(row.get("title", ""))
        if query_lower in title.lower():
            genres_raw = row.get("genres", "")
            if isinstance(genres_raw, str):
                genres = [g for g in genres_raw.split("|") if g and g != "(no genres listed)"]
            else:
                genres = genres_raw if isinstance(genres_raw, list) else []

            year_match = _YEAR_RE.search(title)
            year = int(year_match.group(1)) if year_match else 0

            matches.append(MovieSearchResult(
                movie_id=int(row["movieId"]),
                title=title,
                genres=genres,
                year=year,
            ))
            if len(matches) >= limit:
                break

    return MovieSearchResponse(query=q, results=matches, total=len(matches))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health(
    model: NeuralCollaborativeFiltering | None = Depends(get_model),
    splits: DataSplits | None = Depends(get_splits),
    pool: asyncpg.Pool | None = Depends(get_db_pool),
    orchestrator: OrchestratorAgent = Depends(get_orchestrator),
) -> HealthResponse:
    """Detailed health check with model, data, and agent status."""
    agents: list[AgentHealth] = []
    for agent in [
        orchestrator.user_profiler,
        orchestrator.content_analyzer,
        orchestrator.recsys_engine,
    ]:
        has_real = getattr(agent, "has_real_data", False)
        agents.append(AgentHealth(
            name=agent.agent_name,
            status="ok",
            has_real_data=has_real,
        ))

    data_loaded = splits is not None or pool is not None
    overall = "ok" if model is not None and data_loaded else "degraded"

    # Get counts from DB or splits.
    num_users = 0
    num_movies = 0
    if pool is not None:
        try:
            from src.data.database import fetch_movie_count, fetch_user_count
            num_users = await fetch_user_count(pool)
            num_movies = await fetch_movie_count(pool)
        except Exception:
            logger.warning("Failed to fetch counts from DB", exc_info=True)
    elif splits is not None:
        num_users = splits.num_users
        num_movies = splits.num_movies

    return HealthResponse(
        status=overall,
        model_loaded=model is not None,
        data_loaded=data_loaded,
        num_users=num_users,
        num_movies=num_movies,
        agents=agents,
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


@router.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(
        content=metrics_collector.prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
