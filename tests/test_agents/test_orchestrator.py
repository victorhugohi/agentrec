"""Tests for the orchestrator agent."""

import pytest

from src.agents.orchestrator import OrchestratorAgent


@pytest.mark.asyncio
async def test_orchestrator_process(orchestrator: OrchestratorAgent) -> None:
    """Orchestrator returns expected response structure."""
    result = await orchestrator.run({"user_id": 1, "top_k": 5})

    assert result["user_id"] == 1
    assert "recommendations" in result
    assert isinstance(result["recommendations"], list)
    assert result["metadata"]["top_k"] == 5


@pytest.mark.asyncio
async def test_orchestrator_plan_included(orchestrator: OrchestratorAgent) -> None:
    """Orchestrator response includes the execution plan."""
    result = await orchestrator.run({"user_id": 1, "top_k": 3})

    plan = result["metadata"]["plan"]
    assert len(plan) == 3
    agent_names = [step["agent"] for step in plan]
    assert "user_profiler" in agent_names
    assert "content_analyzer" in agent_names
    assert "recsys_engine" in agent_names


@pytest.mark.asyncio
async def test_orchestrator_caches_results(orchestrator: OrchestratorAgent) -> None:
    """Second call with same payload is served from memory cache."""
    payload = {"user_id": 42, "top_k": 5}
    first = await orchestrator.run(payload)
    second = await orchestrator.run(payload)

    assert first == second
    assert len(orchestrator._memory) == 1


@pytest.mark.asyncio
async def test_orchestrator_default_top_k(orchestrator: OrchestratorAgent) -> None:
    """Orchestrator defaults to top_k=10 when not specified."""
    result = await orchestrator.run({"user_id": 1})

    assert result["metadata"]["top_k"] == 10
