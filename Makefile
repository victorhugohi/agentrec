.PHONY: setup up down logs test train shell seed build migrate help

# Default .env if not present
.env:
	cp .env.example .env

# ── Primary targets ───────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: .env build seed ## Build images, create tables, and seed the database
	@echo "Setup complete. Run 'make up' to start the stack."

build: ## Build all Docker images
	docker compose build

up: .env ## Start all services in the background
	docker compose up -d app db redis
	@echo "Stack running — API at http://localhost:$${API_PORT:-8000}"
	@echo "Health: http://localhost:$${API_PORT:-8000}/api/v1/health"

down: ## Stop and remove all containers
	docker compose down

logs: ## Tail logs for all services
	docker compose logs -f

seed: .env ## Run the database seeder (one-shot)
	docker compose up -d db
	@echo "Waiting for database to be healthy…"
	@docker compose exec db sh -c 'until pg_isready -U postgres; do sleep 1; done' 2>/dev/null
	docker compose run --rm seed
	@echo "Seed complete."

migrate: .env ## Run Alembic migrations
	docker compose run --rm app alembic upgrade head

test: ## Run pytest inside the app container
	docker compose run --rm app python -m pytest tests/ -v --tb=short

train: ## Run the training script inside the app container
	docker compose run --rm \
		-v $$(pwd)/models/checkpoints:/app/models/checkpoints \
		-v $$(pwd)/data:/app/data \
		app python scripts/train_model.py

shell: ## Open a bash shell in the app container
	docker compose run --rm app bash

# ── Convenience ───────────────────────────────────────────────────────

clean: ## Remove containers, volumes, and built images
	docker compose down -v --rmi local
