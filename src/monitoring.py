"""Production observability: Prometheus metrics, structured logging, and cache tracking.

Provides a singleton ``ObservabilityHub`` that collects all application metrics
and exposes them in Prometheus exposition format.  Also configures structured
JSON logging via structlog with request-ID correlation.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from contextvars import ContextVar

import structlog

# ---------------------------------------------------------------------------
# Request-ID context (shared with middleware)
# ---------------------------------------------------------------------------

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# ---------------------------------------------------------------------------
# Structlog configuration
# ---------------------------------------------------------------------------


def configure_logging(*, json: bool = True, level: str = "INFO") -> None:
    """Set up structlog with JSON or console output.

    Args:
        json: If ``True`` (default) emit JSON lines; otherwise use coloured
            console output for local development.
        level: Root log level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(**initial_binds: str | int | float | bool) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger with optional initial context bindings.

    Args:
        **initial_binds: Key-value pairs bound to every log entry from this
            logger (e.g. ``agent="user_profiler"``).

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(**initial_binds)


# ---------------------------------------------------------------------------
# Prometheus metrics hub
# ---------------------------------------------------------------------------

# Histogram bucket boundaries (seconds).
_LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


class _Histogram:
    """A simple Prometheus-style histogram."""

    def __init__(self, buckets: tuple[float, ...] = _LATENCY_BUCKETS) -> None:
        self.buckets = buckets
        self.bucket_counts: dict[str, dict[str, int]] = {}
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def observe(self, labels: str, value: float) -> None:
        """Record an observation."""
        if labels not in self.bucket_counts:
            self.bucket_counts[labels] = {str(b): 0 for b in self.buckets}
            self.bucket_counts[labels]["+Inf"] = 0
        for b in self.buckets:
            if value <= b:
                self.bucket_counts[labels][str(b)] += 1
        self.bucket_counts[labels]["+Inf"] += 1
        self.sums[labels] += value
        self.counts[labels] += 1

    def render(self, name: str, help_text: str) -> list[str]:
        """Render as Prometheus exposition lines."""
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        for labels in sorted(self.bucket_counts):
            buckets = self.bucket_counts[labels]
            cumulative = 0
            for b in self.buckets:
                cumulative += buckets[str(b)]
                lines.append(f'{name}_bucket{{{labels},le="{b}"}} {cumulative}')
            lines.append(f'{name}_bucket{{{labels},le="+Inf"}} {buckets["+Inf"]}')
            lines.append(f"{name}_sum{{{labels}}} {self.sums[labels]:.6f}")
            lines.append(f"{name}_count{{{labels}}} {self.counts[labels]}")
        return lines


class ObservabilityHub:
    """Singleton collecting all application metrics.

    Tracks request counts, latency histograms, model inference time,
    active recommendation gauge, and cache hit/miss ratio.

    Attributes:
        request_count: Counter keyed by method and path.
        request_latency: Histogram of request durations in seconds.
        model_inference: Histogram of model inference durations in seconds.
        active_recommendations: Gauge of in-flight recommendation requests.
        cache_hits: Total cache hits.
        cache_misses: Total cache misses.
    """

    def __init__(self) -> None:
        # Counters
        self.request_count: dict[str, int] = defaultdict(int)
        self.request_errors: dict[str, int] = defaultdict(int)

        # Histograms
        self.request_latency = _Histogram()
        self.model_inference = _Histogram()

        # Gauge
        self.active_recommendations: int = 0

        # Cache
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    # -- Recording methods ---------------------------------------------------

    def record_request(self, method: str, path: str, status: int, duration: float) -> None:
        """Record a completed HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: Normalised request path.
            status: HTTP response status code.
            duration: Request duration in seconds.
        """
        labels = f'method="{method}",path="{path}"'
        self.request_count[labels] += 1
        if status >= 400:
            self.request_errors[labels] += 1
        self.request_latency.observe(labels, duration)

    def record_model_inference(self, duration: float) -> None:
        """Record a model inference duration in seconds."""
        self.model_inference.observe("", duration)

    def inc_active_recommendations(self) -> None:
        """Increment the active recommendations gauge."""
        self.active_recommendations += 1

    def dec_active_recommendations(self) -> None:
        """Decrement the active recommendations gauge."""
        self.active_recommendations = max(0, self.active_recommendations - 1)

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

    @property
    def cache_hit_ratio(self) -> float:
        """Return the cache hit ratio (0.0–1.0)."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    # -- Prometheus exposition -----------------------------------------------

    def prometheus_text(self) -> str:
        """Render all metrics in Prometheus exposition format.

        Returns:
            Multi-line string ready for a ``text/plain`` response.
        """
        lines: list[str] = []

        # Request count
        lines.append("# HELP agentrec_requests_total Total HTTP requests.")
        lines.append("# TYPE agentrec_requests_total counter")
        for labels, count in sorted(self.request_count.items()):
            lines.append(f"agentrec_requests_total{{{labels}}} {count}")

        # Error count
        lines.append("# HELP agentrec_request_errors_total HTTP errors (4xx/5xx).")
        lines.append("# TYPE agentrec_request_errors_total counter")
        for labels, count in sorted(self.request_errors.items()):
            lines.append(f"agentrec_request_errors_total{{{labels}}} {count}")

        # Request latency histogram
        lines.extend(
            self.request_latency.render(
                "agentrec_request_latency_seconds",
                "HTTP request latency in seconds.",
            )
        )

        # Model inference histogram
        lines.extend(
            self.model_inference.render(
                "agentrec_model_inference_seconds",
                "Model inference latency in seconds.",
            )
        )

        # Active recommendations gauge
        lines.append("# HELP agentrec_active_recommendations In-flight recommendation requests.")
        lines.append("# TYPE agentrec_active_recommendations gauge")
        lines.append(f"agentrec_active_recommendations {self.active_recommendations}")

        # Cache hit ratio
        lines.append("# HELP agentrec_cache_hits_total Total cache hits.")
        lines.append("# TYPE agentrec_cache_hits_total counter")
        lines.append(f"agentrec_cache_hits_total {self.cache_hits}")

        lines.append("# HELP agentrec_cache_misses_total Total cache misses.")
        lines.append("# TYPE agentrec_cache_misses_total counter")
        lines.append(f"agentrec_cache_misses_total {self.cache_misses}")

        lines.append("# HELP agentrec_cache_hit_ratio Cache hit ratio (0-1).")
        lines.append("# TYPE agentrec_cache_hit_ratio gauge")
        lines.append(f"agentrec_cache_hit_ratio {self.cache_hit_ratio:.4f}")

        return "\n".join(lines) + "\n"


# Singleton instance.
hub = ObservabilityHub()


# ---------------------------------------------------------------------------
# Timer context manager for model inference
# ---------------------------------------------------------------------------


class inference_timer:  # noqa: N801
    """Context manager that records model inference duration to the hub.

    Usage::

        with inference_timer():
            scores = model.predict_batch(users, items)
    """

    def __enter__(self) -> inference_timer:
        self._start = time.perf_counter()
        hub.inc_active_recommendations()
        return self

    def __exit__(self, *args: object) -> None:
        duration = time.perf_counter() - self._start
        hub.record_model_inference(duration)
        hub.dec_active_recommendations()
