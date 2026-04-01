"""Alembic environment configuration.

Reads the database URL from the ``AGENTREC_DATABASE_URL`` environment
variable, falling back to the project's default Settings value.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context

# Alembic Config object — provides access to alembic.ini values.
config = context.config

# Set up Python loggers from alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# No SQLAlchemy MetaData — we use raw SQL migrations.
target_metadata = None


def _get_url() -> str:
    """Resolve the database URL from env or project settings."""
    url = os.environ.get("AGENTREC_DATABASE_URL")
    if url:
        return url
    # Fall back to alembic.ini value.
    return config.get_main_option("sqlalchemy.url", "")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    from sqlalchemy import create_engine

    url = _get_url()
    connectable = create_engine(url)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
