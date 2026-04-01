"""Request middleware: tracing (request ID) and metrics collection."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from contextvars import ContextVar
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# ContextVar so any code in the request path can read the current request ID.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request/response.

    If the caller supplies an ``X-Request-ID`` header it is reused;
    otherwise a new UUID-4 is generated.  The ID is stored in a
    :class:`~contextvars.ContextVar` so downstream code (e.g. error
    handlers) can include it in structured responses.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        rid = request.headers.get("x-request-id", str(uuid.uuid4()))
        request_id_var.set(rid)
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """In-process counters and histograms for Prometheus-compatible /metrics.

    Thread-safe enough for single-worker uvicorn.  For multi-worker
    deployments, replace with a shared metrics backend (e.g.
    prometheus_client multiprocess mode).
    """

    # Latency histogram bucket boundaries (seconds).
    BUCKETS: tuple[float, ...] = (
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    )

    def __init__(self) -> None:
        self.request_count: dict[str, int] = defaultdict(int)
        self.request_errors: dict[str, int] = defaultdict(int)
        self.latency_buckets: dict[str, dict[str, int]] = {}
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.latency_count: dict[str, int] = defaultdict(int)
        self.model_inference_sum: float = 0.0
        self.model_inference_count: int = 0

    def record_request(
        self, method: str, path: str, status: int, duration: float
    ) -> None:
        """Record a completed HTTP request."""
        key = f"{method} {path}"
        self.request_count[key] += 1
        if status >= 400:
            self.request_errors[key] += 1
        self._observe_latency(key, duration)

    def record_model_inference(self, duration: float) -> None:
        """Record a model inference duration."""
        self.model_inference_sum += duration
        self.model_inference_count += 1

    def _observe_latency(self, key: str, duration: float) -> None:
        """Insert *duration* into the histogram buckets for *key*."""
        if key not in self.latency_buckets:
            self.latency_buckets[key] = {str(b): 0 for b in self.BUCKETS}
            self.latency_buckets[key]["+Inf"] = 0
        for b in self.BUCKETS:
            if duration <= b:
                self.latency_buckets[key][str(b)] += 1
        self.latency_buckets[key]["+Inf"] += 1
        self.latency_sum[key] += duration
        self.latency_count[key] += 1

    def prometheus_text(self) -> str:
        """Render all metrics in Prometheus exposition format."""
        lines: list[str] = []

        # Request count
        lines.append("# HELP agentrec_requests_total Total HTTP requests.")
        lines.append("# TYPE agentrec_requests_total counter")
        for key, count in sorted(self.request_count.items()):
            method, path = key.split(" ", 1)
            lines.append(
                f'agentrec_requests_total{{method="{method}",path="{path}"}} {count}'
            )

        # Error count
        lines.append("# HELP agentrec_request_errors_total HTTP request errors (4xx/5xx).")
        lines.append("# TYPE agentrec_request_errors_total counter")
        for key, count in sorted(self.request_errors.items()):
            method, path = key.split(" ", 1)
            lines.append(
                f'agentrec_request_errors_total{{method="{method}",path="{path}"}} {count}'
            )

        # Latency histogram
        lines.append("# HELP agentrec_request_duration_seconds Request latency.")
        lines.append("# TYPE agentrec_request_duration_seconds histogram")
        for key in sorted(self.latency_buckets):
            method, path = key.split(" ", 1)
            label = f'method="{method}",path="{path}"'
            buckets = self.latency_buckets[key]
            cumulative = 0
            for b in self.BUCKETS:
                cumulative += buckets[str(b)]
                lines.append(
                    f"agentrec_request_duration_seconds_bucket{{{label},le=\"{b}\"}} {cumulative}"
                )
            cumulative += buckets["+Inf"] - sum(
                buckets[str(b)] for b in self.BUCKETS
            )
            lines.append(
                f'agentrec_request_duration_seconds_bucket{{{label},le="+Inf"}} {buckets["+Inf"]}'
            )
            lines.append(
                f"agentrec_request_duration_seconds_sum{{{label}}} {self.latency_sum[key]:.6f}"
            )
            lines.append(
                f"agentrec_request_duration_seconds_count{{{label}}} {self.latency_count[key]}"
            )

        # Model inference
        lines.append("# HELP agentrec_model_inference_seconds Model inference latency.")
        lines.append("# TYPE agentrec_model_inference_seconds summary")
        lines.append(
            f"agentrec_model_inference_seconds_sum {self.model_inference_sum:.6f}"
        )
        lines.append(
            f"agentrec_model_inference_seconds_count {self.model_inference_count}"
        )

        return "\n".join(lines) + "\n"


# Singleton instance used by the metrics middleware and /metrics endpoint.
metrics_collector = MetricsCollector()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for every request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        # Normalise path to avoid high-cardinality labels from path params.
        path = self._normalise_path(request.url.path)
        metrics_collector.record_request(
            request.method, path, response.status_code, duration,
        )
        return response

    @staticmethod
    def _normalise_path(path: str) -> str:
        """Replace numeric path segments with ``:id`` to limit cardinality."""
        parts = path.rstrip("/").split("/")
        normalised = []
        for part in parts:
            if part.isdigit():
                normalised.append(":id")
            else:
                normalised.append(part)
        return "/".join(normalised)
