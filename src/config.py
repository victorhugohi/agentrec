"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings.

    Attributes:
        app_name: Display name of the application.
        debug: Enable debug mode.
        database_url: PostgreSQL connection string.
        redis_url: Redis connection string.
        model_path: Filesystem path to the trained NCF model weights.
        movielens_path: Filesystem path to the MovieLens dataset.
        embedding_dim: Dimensionality of user/item embeddings in the NCF model.
        batch_size: Training and inference batch size.
        top_k: Default number of recommendations to return.
    """

    app_name: str = "AgentRec"
    debug: bool = False

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/agentrec"
    redis_url: str = "redis://localhost:6379/0"

    # Model
    model_path: str = "models/checkpoints/ncf_best.pt"
    movielens_variant: str = "small"
    data_dir: str = "data"
    embedding_dim: int = 64
    batch_size: int = 256
    top_k: int = 10

    # CORS
    cors_origins: list[str] = ["*"]

    model_config = {"env_prefix": "AGENTREC_"}


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings.

    Returns:
        The singleton Settings instance.
    """
    return Settings()
