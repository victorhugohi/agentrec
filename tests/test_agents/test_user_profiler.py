"""Tests for the user profiler agent."""

import pytest

from src.agents.user_profiler import UserProfilerAgent


@pytest.mark.asyncio
async def test_profiler_known_user(user_profiler: UserProfilerAgent) -> None:
    """Profiler returns ratings and embedding for a known user."""
    result = await user_profiler.run({"user_id": 1})

    assert result["user_id"] == 1
    assert result["rating_count"] == 5
    assert len(result["ratings"]) == 5
    assert isinstance(result["embedding"], list)
    assert len(result["embedding"]) == 64


@pytest.mark.asyncio
async def test_profiler_unknown_user(user_profiler: UserProfilerAgent) -> None:
    """Profiler returns empty profile for an unknown user."""
    result = await user_profiler.run({"user_id": 9999})

    assert result["user_id"] == 9999
    assert result["rating_count"] == 0
    assert result["ratings"] == []
    assert result["embedding"] == [0.0] * 64
    assert result["avg_rating"] == 0.0


@pytest.mark.asyncio
async def test_profiler_embedding_uses_high_ratings(user_profiler: UserProfilerAgent) -> None:
    """Embedding is computed only from items rated >= 3.5."""
    result = await user_profiler.run({"user_id": 1})

    # User 1 has 4 items rated >= 3.5 (ids: 1, 50, 110, 296) and 1 below (id: 260)
    # The embedding should NOT include item 260's vector
    assert result["embedding"] != [0.0] * 64
    # With 4 high-rated items the avg_rating should be > 3.5
    assert result["avg_rating"] > 3.0


@pytest.mark.asyncio
async def test_profiler_avg_rating(user_profiler: UserProfilerAgent) -> None:
    """avg_rating is the mean of all ratings, not just high ones."""
    result = await user_profiler.run({"user_id": 2})

    # User 2: ratings are 3.0, 4.0, 5.0, 3.5 → mean = 3.875
    assert result["avg_rating"] == pytest.approx(3.88, abs=0.01)


@pytest.mark.asyncio
async def test_profiler_agent_name(user_profiler: UserProfilerAgent) -> None:
    """Agent name is 'user_profiler'."""
    assert user_profiler.agent_name == "user_profiler"
