"""Tests for the content analyzer agent."""

import pytest

from src.agents.content_analyzer import ContentAnalyzerAgent


@pytest.mark.asyncio
async def test_analyzer_returns_full_catalog(content_analyzer: ContentAnalyzerAgent) -> None:
    """Without specific item_ids, returns the entire catalog."""
    result = await content_analyzer.run({"user_id": 1})

    assert result["user_id"] == 1
    assert len(result["items"]) == 10  # full mock catalog size


@pytest.mark.asyncio
async def test_analyzer_filters_by_item_ids(content_analyzer: ContentAnalyzerAgent) -> None:
    """With specific item_ids, returns only those items."""
    result = await content_analyzer.run({"user_id": 1, "item_ids": [1, 50, 296]})

    returned_ids = {item["item_id"] for item in result["items"]}
    assert returned_ids == {1, 50, 296}
    assert len(result["items"]) == 3


@pytest.mark.asyncio
async def test_analyzer_items_have_embeddings(content_analyzer: ContentAnalyzerAgent) -> None:
    """Every returned item has a 64-dimensional embedding from the shared store."""
    result = await content_analyzer.run({"user_id": 1, "item_ids": [47]})

    item = result["items"][0]
    assert item["item_id"] == 47
    assert len(item["embedding"]) == 64
    assert item["title"] == "Seven"
    assert "Crime" in item["genres"]


@pytest.mark.asyncio
async def test_analyzer_items_have_metadata(content_analyzer: ContentAnalyzerAgent) -> None:
    """Returned items include title, genres, year, and tags."""
    result = await content_analyzer.run({"user_id": 1, "item_ids": [110]})

    item = result["items"][0]
    assert item["title"] == "Braveheart"
    assert item["year"] == 1995
    assert "epic" in item["tags"]
    assert "Drama" in item["genres"]


@pytest.mark.asyncio
async def test_analyzer_unknown_item_ids_skipped(content_analyzer: ContentAnalyzerAgent) -> None:
    """Unknown item IDs are silently skipped."""
    result = await content_analyzer.run({"user_id": 1, "item_ids": [1, 99999]})

    assert len(result["items"]) == 1
    assert result["items"][0]["item_id"] == 1


@pytest.mark.asyncio
async def test_analyzer_agent_name(content_analyzer: ContentAnalyzerAgent) -> None:
    """Agent name is 'content_analyzer'."""
    assert content_analyzer.agent_name == "content_analyzer"
