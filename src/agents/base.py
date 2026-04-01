"""Abstract base class for all agents."""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class that all agents must inherit from.

    Provides a common interface, logging, and an in-memory result cache
    so repeated calls with the same payload are served instantly.

    Attributes:
        _name: Internal agent identifier.
        _memory: Dict-based cache mapping payload hashes to prior results.
    """

    def __init__(self, name: str) -> None:
        """Initialize the agent.

        Args:
            name: Human-readable agent identifier.
        """
        self._name = name
        self._memory: dict[str, dict[str, Any]] = {}

    @property
    def agent_name(self) -> str:
        """Return the agent's identifier."""
        return self._name

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent with caching and logging.

        Checks the memory cache first. On a miss, delegates to
        :meth:`process`, stores the result, and returns it.

        Args:
            payload: Arbitrary request data for this agent.

        Returns:
            A dict containing the agent's response.
        """
        cache_key = self._cache_key(payload)

        if cache_key in self._memory:
            logger.debug("%s: cache hit for %s", self.agent_name, cache_key[:12])
            return self._memory[cache_key]

        logger.info("%s: processing request", self.agent_name)
        result = await self.process(payload)
        self._memory[cache_key] = result
        logger.info("%s: done", self.agent_name)
        return result

    @abstractmethod
    async def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Process an incoming request and return a result.

        Subclasses must implement this with their domain logic.

        Args:
            payload: Arbitrary request data for this agent.

        Returns:
            A dict containing the agent's response.
        """
        ...

    def clear_memory(self) -> None:
        """Clear the in-memory result cache."""
        self._memory.clear()

    def _cache_key(self, payload: dict[str, Any]) -> str:
        """Compute a deterministic hash for a payload dict.

        Args:
            payload: The request data to hash.

        Returns:
            A hex digest string suitable as a cache key.
        """
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_name={self.agent_name!r})"
