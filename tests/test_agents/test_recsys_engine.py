"""Tests for the recommendation engine agent."""

import pytest

from src.agents.recsys_engine import RecsysEngineAgent
from src.agents.user_profiler import UserProfilerAgent
from src.agents.content_analyzer import ContentAnalyzerAgent


@pytest.mark.asyncio
async def test_engine_returns_scored_items(
    user_profiler: UserProfilerAgent,
    content_analyzer: ContentAnalyzerAgent,
    recsys_engine: RecsysEngineAgent,
) -> None:
    """Engine returns scored items sorted by descending score."""
    profile = await user_profiler.run({"user_id": 1})
    content = await content_analyzer.run({"user_id": 1})

    result = await recsys_engine.run({
        "user_profile": profile,
        "content_features": content,
        "top_k": 5,
    })

    ranked = result["ranked_items"]
    assert len(ranked) == 5
    # Verify descending score order
    scores = [item["score"] for item in ranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_engine_scores_in_valid_range(
    user_profiler: UserProfilerAgent,
    content_analyzer: ContentAnalyzerAgent,
    recsys_engine: RecsysEngineAgent,
) -> None:
    """All scores should be between 0 and 1 (sigmoid output)."""
    profile = await user_profiler.run({"user_id": 1})
    content = await content_analyzer.run({"user_id": 1})

    result = await recsys_engine.run({
        "user_profile": profile,
        "content_features": content,
        "top_k": 10,
    })

    for item in result["ranked_items"]:
        assert 0.0 <= item["score"] <= 1.0


@pytest.mark.asyncio
async def test_engine_respects_top_k(
    user_profiler: UserProfilerAgent,
    content_analyzer: ContentAnalyzerAgent,
    recsys_engine: RecsysEngineAgent,
) -> None:
    """Engine returns at most top_k items."""
    profile = await user_profiler.run({"user_id": 1})
    content = await content_analyzer.run({"user_id": 1})

    result = await recsys_engine.run({
        "user_profile": profile,
        "content_features": content,
        "top_k": 3,
    })

    assert len(result["ranked_items"]) == 3


@pytest.mark.asyncio
async def test_engine_items_have_metadata(
    user_profiler: UserProfilerAgent,
    content_analyzer: ContentAnalyzerAgent,
    recsys_engine: RecsysEngineAgent,
) -> None:
    """Scored items include title and genres from the content analyzer."""
    profile = await user_profiler.run({"user_id": 1})
    content = await content_analyzer.run({"user_id": 1})

    result = await recsys_engine.run({
        "user_profile": profile,
        "content_features": content,
        "top_k": 10,
    })

    for item in result["ranked_items"]:
        assert item["title"] != ""
        assert isinstance(item["genres"], list)


@pytest.mark.asyncio
async def test_engine_empty_candidates(
    user_profiler: UserProfilerAgent,
    recsys_engine: RecsysEngineAgent,
) -> None:
    """Engine handles empty candidate list gracefully."""
    profile = await user_profiler.run({"user_id": 1})
    empty_content = {"user_id": 1, "items": []}

    result = await recsys_engine.run({
        "user_profile": profile,
        "content_features": empty_content,
        "top_k": 5,
    })

    assert result["ranked_items"] == []


@pytest.mark.asyncio
async def test_engine_agent_name(recsys_engine: RecsysEngineAgent) -> None:
    """Agent name is 'recsys_engine'."""
    assert recsys_engine.agent_name == "recsys_engine"
