"""Tests for the BaseAgent abstract class."""

from typing import Any

import pytest

from src.agents.base import BaseAgent


class _StubAgent(BaseAgent):
    """Minimal concrete agent for testing the base class."""

    def __init__(self) -> None:
        super().__init__(name="stub")
        self.call_count = 0

    async def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        return {"echo": payload}


@pytest.fixture
def stub_agent() -> _StubAgent:
    return _StubAgent()


def test_agent_name(stub_agent: _StubAgent) -> None:
    """agent_name property returns the name passed to __init__."""
    assert stub_agent.agent_name == "stub"


def test_repr(stub_agent: _StubAgent) -> None:
    """repr includes class name and agent_name."""
    assert repr(stub_agent) == "_StubAgent(agent_name='stub')"


@pytest.mark.asyncio
async def test_run_delegates_to_process(stub_agent: _StubAgent) -> None:
    """run() calls process() and returns its result."""
    result = await stub_agent.run({"key": "value"})
    assert result == {"echo": {"key": "value"}}
    assert stub_agent.call_count == 1


@pytest.mark.asyncio
async def test_run_caches_on_second_call(stub_agent: _StubAgent) -> None:
    """run() serves from cache on duplicate payloads."""
    payload = {"x": 1}
    await stub_agent.run(payload)
    await stub_agent.run(payload)

    assert stub_agent.call_count == 1
    assert len(stub_agent._memory) == 1


@pytest.mark.asyncio
async def test_run_different_payloads_not_cached(stub_agent: _StubAgent) -> None:
    """run() calls process() again for different payloads."""
    await stub_agent.run({"a": 1})
    await stub_agent.run({"b": 2})

    assert stub_agent.call_count == 2
    assert len(stub_agent._memory) == 2


@pytest.mark.asyncio
async def test_clear_memory(stub_agent: _StubAgent) -> None:
    """clear_memory() empties the cache."""
    await stub_agent.run({"x": 1})
    assert len(stub_agent._memory) == 1

    stub_agent.clear_memory()
    assert len(stub_agent._memory) == 0

    # Next call should hit process() again
    await stub_agent.run({"x": 1})
    assert stub_agent.call_count == 2
