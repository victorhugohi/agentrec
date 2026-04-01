"""Orchestrator agent that coordinates the recommendation pipeline."""

import asyncio
import logging
from typing import Any

from src.agents.base import BaseAgent
from src.agents.user_profiler import UserProfilerAgent
from src.agents.content_analyzer import ContentAnalyzerAgent
from src.agents.recsys_engine import RecsysEngineAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """Coordinates the full recommendation pipeline using plan-and-execute.

    The orchestrator builds an execution plan from the incoming request,
    runs independent steps in parallel where possible, and assembles the
    final result.

    Pipeline:
        1. **Plan** — determine which agents to invoke and in what order.
        2. **Execute parallel** — run user_profiler and content_analyzer
           concurrently via ``asyncio.gather``.
        3. **Execute sequential** — pass both outputs to recsys_engine for
           scoring and ranking.
        4. **Assemble** — build the final response.

    Attributes:
        user_profiler: Agent responsible for building user profiles.
        content_analyzer: Agent responsible for analyzing item content.
        recsys_engine: Agent responsible for scoring and ranking.
    """

    def __init__(
        self,
        user_profiler: UserProfilerAgent,
        content_analyzer: ContentAnalyzerAgent,
        recsys_engine: RecsysEngineAgent,
    ) -> None:
        """Initialize the orchestrator with its sub-agents.

        Args:
            user_profiler: The user profiler agent.
            content_analyzer: The content analyzer agent.
            recsys_engine: The recommendation engine agent.
        """
        super().__init__(name="orchestrator")
        self.user_profiler = user_profiler
        self.content_analyzer = content_analyzer
        self.recsys_engine = recsys_engine

    async def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the full recommendation pipeline for a user.

        Args:
            payload: Must contain ``user_id`` (int) and optionally
                ``top_k`` (int) for the number of results.

        Returns:
            A dict with ``user_id``, ``recommendations`` (list of item dicts),
            and ``metadata`` including the execution plan.
        """
        user_id: int = payload["user_id"]
        top_k: int = payload.get("top_k", 10)

        # --- Plan phase ---
        plan = self._build_plan(user_id, top_k)
        logger.info("Execution plan: %s", [step["agent"] for step in plan])

        # --- Execute phase ---

        # Step 1: Run user_profiler and content_analyzer in parallel.
        # content_analyzer receives user_id so it can independently fetch
        # candidate items while the profiler builds the user embedding.
        user_profile, content_features = await asyncio.gather(
            self.user_profiler.run({"user_id": user_id}),
            self.content_analyzer.run({"user_id": user_id}),
        )

        # Step 2: Run recsys_engine with combined outputs (sequential).
        recommendations = await self.recsys_engine.run({
            "user_profile": user_profile,
            "content_features": content_features,
            "top_k": top_k,
        })

        # --- Assemble phase ---
        return {
            "user_id": user_id,
            "recommendations": recommendations.get("ranked_items", []),
            "metadata": {
                "model": "ncf",
                "top_k": top_k,
                "plan": plan,
            },
        }

    def _build_plan(
        self, user_id: int, top_k: int
    ) -> list[dict[str, Any]]:
        """Build the execution plan describing which agents run and when.

        Args:
            user_id: The target user.
            top_k: Number of recommendations requested.

        Returns:
            An ordered list of plan steps. Each step is a dict with
            ``agent``, ``parallel_group``, and ``inputs``.
        """
        return [
            {
                "agent": self.user_profiler.agent_name,
                "parallel_group": 0,
                "inputs": {"user_id": user_id},
            },
            {
                "agent": self.content_analyzer.agent_name,
                "parallel_group": 0,
                "inputs": {"user_id": user_id},
            },
            {
                "agent": self.recsys_engine.agent_name,
                "parallel_group": 1,
                "inputs": {"top_k": top_k},
            },
        ]
