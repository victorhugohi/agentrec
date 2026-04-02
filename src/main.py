"""FastAPI application entry point.

Configures middleware, loads the trained model and data on startup,
and handles graceful shutdown.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.api.dependencies import build_orchestrator, load_data, load_model
from src.api.routes import router
from src.config import get_settings
from src.monitoring import configure_logging, get_logger, hub, request_id_var

configure_logging(json=True, level="INFO")
logger = get_logger(component="app")


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

# ---------------------------------------------------------------------------
# Middleware (order matters: outermost = first added, but Starlette reverses
# the call order — so the LAST add_middleware wraps the outermost layer).
# ---------------------------------------------------------------------------


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request/response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = request.headers.get("x-request-id", str(uuid.uuid4()))
        request_id_var.set(rid)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=rid)
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, duration, and user-agent."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        log = get_logger(component="http")
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            request_id=request_id_var.get(""),
            user_agent=request.headers.get("user-agent", ""),
        )
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for every request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        path = self._normalise_path(request.url.path)
        hub.record_request(request.method, path, response.status_code, duration)
        return response

    @staticmethod
    def _normalise_path(path: str) -> str:
        """Replace numeric path segments with ``:id`` to limit cardinality."""
        parts = path.rstrip("/").split("/")
        return "/".join(":id" if part.isdigit() else part for part in parts)


settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIdMiddleware)


# --- Global exception handler ---


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured JSON error for unhandled exceptions."""
    rid = request_id_var.get("")
    logger.error("unhandled_error", request_id=rid, error=str(exc), exc_info=True)
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
