# ── Stage 1: Builder ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed by some wheels (e.g. asyncpg).
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual-env so we can copy it cleanly into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (layer caching).
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Install dev dependencies for running tests inside the container.
RUN pip install --no-cache-dir ".[dev]"

# ── Stage 2: Runtime ─────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libpq is needed at runtime by asyncpg; curl for the healthcheck.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --create-home app

# Copy the pre-built virtual-env from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy source code, scripts, and tests.
COPY --chown=app:app src/ src/
COPY --chown=app:app scripts/ scripts/
COPY --chown=app:app tests/ tests/
COPY --chown=app:app alembic/ alembic/
COPY --chown=app:app alembic.ini pyproject.toml ./

# Create directories the app may write to.
RUN mkdir -p models/checkpoints data && chown -R app:app models data

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
