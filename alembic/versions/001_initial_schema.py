"""Initial schema — users, movies, ratings, recommendations_log.

Revision ID: 001
Revises: None
Create Date: 2026-03-31
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id   INTEGER PRIMARY KEY
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            movie_id  INTEGER PRIMARY KEY,
            title     TEXT    NOT NULL DEFAULT '',
            genres    TEXT[]  NOT NULL DEFAULT '{}',
            year      INTEGER NOT NULL DEFAULT 0
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            user_id   INTEGER NOT NULL REFERENCES users(user_id),
            movie_id  INTEGER NOT NULL REFERENCES movies(movie_id),
            rating    REAL    NOT NULL,
            timestamp BIGINT  NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, movie_id, timestamp)
        );
    """)

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id);"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS recommendations_log (
            id         SERIAL PRIMARY KEY,
            user_id    INTEGER   NOT NULL,
            items      JSONB     NOT NULL DEFAULT '[]',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_reclog_user ON recommendations_log(user_id);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS recommendations_log CASCADE;")
    op.execute("DROP TABLE IF EXISTS ratings CASCADE;")
    op.execute("DROP TABLE IF EXISTS movies CASCADE;")
    op.execute("DROP TABLE IF EXISTS users CASCADE;")
