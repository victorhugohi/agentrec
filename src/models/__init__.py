"""ML models and Pydantic schemas."""

from src.models.embeddings import EmbeddingStore, get_embedding_store
from src.models.ncf import NeuralCollaborativeFiltering
from src.models.schemas import (
    ContentAnalyzerInput,
    ContentAnalyzerOutput,
    ContentItem,
    MovieItem,
    Rating,
    RecommendationRequest,
    RecommendationResponse,
    RecsysEngineInput,
    RecsysEngineOutput,
    ScoredItem,
    UserProfile,
    UserProfileInput,
    UserProfileOutput,
)

__all__ = [
    "EmbeddingStore",
    "get_embedding_store",
    "ContentAnalyzerInput",
    "ContentAnalyzerOutput",
    "ContentItem",
    "MovieItem",
    "NeuralCollaborativeFiltering",
    "Rating",
    "RecommendationRequest",
    "RecommendationResponse",
    "RecsysEngineInput",
    "RecsysEngineOutput",
    "ScoredItem",
    "UserProfile",
    "UserProfileInput",
    "UserProfileOutput",
]
