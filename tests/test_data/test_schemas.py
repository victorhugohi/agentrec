"""Tests for data pipeline Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.data.schemas import DataRecommendationRequest, Movie, Rating, UserHistory


class TestRating:
    def test_valid_rating(self) -> None:
        r = Rating(user_id=0, movie_id=5, rating=0.8, timestamp=1000000)
        assert r.user_id == 0
        assert r.movie_id == 5
        assert r.rating == 0.8
        assert r.timestamp == 1000000

    def test_normalised_rating_bounds(self) -> None:
        low = Rating(user_id=0, movie_id=0, rating=0.0, timestamp=0)
        high = Rating(user_id=0, movie_id=0, rating=1.0, timestamp=0)
        assert low.rating == 0.0
        assert high.rating == 1.0


class TestMovie:
    def test_movie_with_genres(self) -> None:
        m = Movie(movie_id=0, title="Toy Story (1995)", genres=["Animation", "Comedy"])
        assert m.title == "Toy Story (1995)"
        assert len(m.genres) == 2

    def test_movie_default_genres(self) -> None:
        m = Movie(movie_id=0, title="Unknown")
        assert m.genres == []


class TestUserHistory:
    def test_user_with_ratings(self) -> None:
        ratings = [Rating(user_id=0, movie_id=i, rating=0.5, timestamp=i) for i in range(5)]
        h = UserHistory(user_id=0, ratings=ratings)
        assert h.user_id == 0
        assert len(h.ratings) == 5

    def test_empty_history(self) -> None:
        h = UserHistory(user_id=99, ratings=[])
        assert len(h.ratings) == 0


class TestDataRecommendationRequest:
    def test_defaults(self) -> None:
        req = DataRecommendationRequest(user_id=0)
        assert req.n_recommendations == 10
        assert req.exclude_seen is True

    def test_custom_values(self) -> None:
        req = DataRecommendationRequest(user_id=5, n_recommendations=20, exclude_seen=False)
        assert req.n_recommendations == 20
        assert req.exclude_seen is False

    def test_validation_min(self) -> None:
        with pytest.raises(ValidationError):
            DataRecommendationRequest(user_id=0, n_recommendations=0)

    def test_validation_max(self) -> None:
        with pytest.raises(ValidationError):
            DataRecommendationRequest(user_id=0, n_recommendations=501)
