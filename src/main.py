"""FastAPI application entry point.

Configures middleware, loads the trained model and data on startup,
and handles graceful shutdown.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import build_orchestrator, load_data, load_model
from src.api.middleware import MetricsMiddleware, RequestIdMiddleware, request_id_var
from src.api.routes import router
from src.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info("Starting %s (debug=%s)", settings.app_name, settings.debug)

    # Load model and data (both optional — app degrades gracefully).
    load_model()
    load_data()

    # Initialise infrastructure (DB and cache fail open).
    db_ok = await _try_init_db(settings.database_url)
    cache_ok = await _try_init_cache(settings.redis_url)
    logger.info("Infrastructure: db=%s, cache=%s", db_ok, cache_ok)

    # Build orchestrator after DB is initialised so agents can use it.
    build_orchestrator()

    yield

    # Shutdown
    logger.info("Shutting down %s …", settings.app_name)
    await _try_close_cache()
    await _try_close_db()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Infrastructure helpers (fail open — the API still works without DB/Redis)
# ---------------------------------------------------------------------------


async def _try_init_db(url: str) -> bool:
    try:
        from src.data.database import init_db
        await init_db(url)
        return True
    except Exception:
        logger.warning("Database unavailable — running without DB", exc_info=True)
        return False


async def _try_close_db() -> None:
    try:
        from src.data.database import close_db
        await close_db()
    except Exception:
        pass


async def _try_init_cache(url: str) -> bool:
    try:
        from src.data.cache import init_cache
        await init_cache(url)
        return True
    except Exception:
        logger.warning("Redis unavailable — running without cache", exc_info=True)
        return False


async def _try_close_cache() -> None:
    try:
        from src.data.cache import close_cache
        await close_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgentRec",
    description="Multi-agent recommendation engine using Neural Collaborative Filtering.",
    version="0.1.0",
    lifespan=lifespan,
)

# --- Middleware (order matters: outermost first) ---
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)
app.add_middleware(RequestIdMiddleware)


# --- Global exception handler ---


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured JSON error for unhandled exceptions."""
    rid = request_id_var.get("")
    logger.error("Unhandled error [%s]: %s", rid, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred.",
            "request_id": rid,
        },
    )


# --- Routes ---
app.include_router(router, prefix="/api/v1")
