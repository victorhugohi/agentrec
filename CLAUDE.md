# AgentRec

Multi-agent recommendation engine using Neural Collaborative Filtering on MovieLens data.

## Tech Stack

- **Language:** Python 3.11
- **Web framework:** FastAPI
- **ML framework:** PyTorch
- **Database:** PostgreSQL
- **Cache:** Redis
- **Containerization:** Docker

## Architecture

Four agents collaborate to produce recommendations:

- **orchestrator** — coordinates the recommendation pipeline across agents
- **user_profiler** — builds and maintains user preference profiles
- **content_analyzer** — analyzes item features and metadata
- **recsys_engine** — runs the Neural Collaborative Filtering model for scoring/ranking

## Project Structure

- All source code lives in `src/`
- Dependencies managed via `pyproject.toml` (not requirements.txt)

## Code Style

- Type hints on all function signatures and variables where non-obvious
- Google-style docstrings on all public functions/classes
- Format with `ruff format`, not `black`
- Before committing, always run `/lint`
- All code must pass `ruff`, `mypy`, and maintain 80% test coverage

### Quality Tools

- **Linting:** `ruff check src/ tests/` (rules: E, F, I, UP, B, SIM, ANN)
- **Type checking:** `mypy src/` (strict mode, ignores tests/)
- **Formatting:** `ruff format src/ tests/`
- **Coverage:** `pytest --cov=src --cov-fail-under=80`
- **Pre-commit:** `pre-commit run --all-files`

## Testing

- Framework: **pytest**
- Tests go in `tests/` mirroring the `src/` structure
- Run tests: `pytest`

## Training

Train the NCF model on MovieLens data:

```bash
# Train on MovieLens-small (100K ratings, fast iteration)
python scripts/train_model.py --variant small --epochs 30 --batch-size 256

# Train on MovieLens-25M (production)
python scripts/train_model.py --variant 25m --epochs 50 --batch-size 1024 --num-workers 4

# Custom hyperparameters
python scripts/train_model.py --embedding-dim 32 --mlp-layers 64,32,16 --lr 0.002 --dropout 0.3
```

Best checkpoint is saved to `models/checkpoints/ncf_best.pt`. Early stopping (patience=5) on validation loss.

### Evaluate

```bash
python scripts/evaluate_model.py
python scripts/evaluate_model.py --checkpoint models/checkpoints/ncf_best.pt --k 20
```

### Expected Metrics (MovieLens-small, default hyperparameters)

| Metric     | Expected Range |
|------------|---------------|
| NDCG@10    | 0.35 – 0.45   |
| HR@10      | 0.55 – 0.70   |
| MRR        | 0.25 – 0.40   |
| Coverage   | 0.40 – 0.60   |

## Docker Deployment

The full stack runs via Docker Compose: API server, PostgreSQL, Redis, and a one-shot database seeder.

### Quick Start

```bash
# First time — build, create tables, seed database
make setup

# Start the stack
make up

# Verify
curl http://localhost:8000/api/v1/health
```

### Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `app` | Multi-stage Python 3.11-slim | 8000 | FastAPI + uvicorn |
| `db` | postgres:16-alpine | 5432 | Ratings, movies, users |
| `redis` | redis:7-alpine | 6379 | Recommendation cache |
| `seed` | Same as app (one-shot) | — | Bulk-loads Parquet → Postgres |

### Make Targets

```bash
make setup      # Build + seed (first time)
make up         # Start services
make down       # Stop services
make logs       # Tail all logs
make test       # Run pytest in container
make train      # Run training script in container
make shell      # Bash into app container
make seed       # Re-seed the database
make migrate    # Run Alembic migrations
make monitoring # Start with Prometheus + Grafana
make clean      # Remove containers, volumes, images
```

### Configuration

Copy `.env.example` to `.env` and edit as needed. All settings use the `AGENTREC_` prefix.

Key variables: `AGENTREC_DATABASE_URL`, `AGENTREC_REDIS_URL`, `AGENTREC_MODEL_PATH`, `AGENTREC_DATA_DIR`, `API_PORT`.

### Database Migrations

```bash
# Run pending migrations
make migrate

# Generate a new migration (local)
alembic revision --autogenerate -m "description"
```

## Observability

Production monitoring via `src/monitoring.py`:

- **Structured logging:** JSON via structlog with request_id correlation
- **Prometheus metrics:** request_count, request_latency_seconds, model_inference_seconds, active_recommendations, cache_hit_ratio
- **Metrics endpoint:** `GET /api/v1/metrics` (Prometheus exposition format)
- **Request logging middleware:** logs method, path, status_code, duration_ms, request_id, user_agent

### Monitoring Stack

```bash
make monitoring   # Starts app + Prometheus + Grafana
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

Grafana is pre-provisioned with an "AgentRec Overview" dashboard showing request rate, latency p50/p95/p99, model inference time, and cache hit ratio.

## CI/CD

GitHub Actions runs two workflows:

### CI (`.github/workflows/ci.yml`)

Triggers on push to `main` and on pull requests. Runs a matrix of Python 3.11 and 3.12 with PostgreSQL and Redis services.

Steps: ruff format check → ruff lint → mypy strict → pytest with 80% coverage gate → Docker build verification.

### Release (`.github/workflows/release.yml`)

Triggers on tags matching `v*`. Builds the Docker image and pushes to GitHub Container Registry (`ghcr.io`) tagged with the version and `latest`.

```bash
# Create a release (or use /release slash command)
# 1. Bump version in pyproject.toml
# 2. Commit and tag
git tag v0.2.0
git push origin v0.2.0
```

## Common Commands

```bash
# Run the API server (local dev)
uvicorn src.main:app --reload

# Run tests
pytest

# Seed the database (local dev)
python scripts/seed_db.py --database-url postgresql://postgres:postgres@localhost:5432/agentrec

# Docker — full stack
make setup && make up
```
