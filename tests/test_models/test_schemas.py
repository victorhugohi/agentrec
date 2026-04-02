"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.models.schemas import MovieItem, RecommendationRequest, RecommendationResponse


def test_recommendation_request_defaults() -> None:
    """RecommendationRequest uses default top_k=10."""
    req = RecommendationRequest(user_id=42)
    assert req.top_k == 10


def test_recommendation_request_validation() -> None:
    """RecommendationRequest rejects top_k < 1."""
    with pytest.raises(ValidationError):
        RecommendationRequest(user_id=1, top_k=0)


def test_recommendation_response_roundtrip() -> None:
    """RecommendationResponse serializes and deserializes correctly."""
    resp = RecommendationResponse(
        user_id=1,
        recommendations=[MovieItem(item_id=10, title="Test Movie", score=0.95)],
    )
    data = resp.model_dump()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["score"] == 0.95
