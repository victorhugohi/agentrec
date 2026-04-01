"""Recommendation engine agent that scores and ranks items using NCF."""

import logging
from typing import Any

import torch

from src.agents.base import BaseAgent
from src.models.embeddings import EmbeddingStore, get_embedding_store
from src.models.ncf import NeuralCollaborativeFiltering
from src.models.schemas import (
    ContentAnalyzerOutput,
    RecsysEngineInput,
    RecsysEngineOutput,
    ScoredItem,
    UserProfileOutput,
)

logger = logging.getLogger(__name__)

# Constants matching the mock data dimensions.
_NUM_USERS = 1000
_NUM_ITEMS = 500
_EMBEDDING_DIM = 64


class RecsysEngineAgent(BaseAgent):
    """Scores and ranks candidate items using the Neural Collaborative Filtering model.

    Takes a user profile and content features, runs them through the NCF
    model's :meth:`predict_batch` method, and returns a ranked list of
    recommendations sorted by predicted relevance score.

    Attributes:
        _model: The NCF model instance used for scoring.
        _embedding_store: Shared embedding store (updated when the model loads).
    """

    def __init__(
        self,
        model: NeuralCollaborativeFiltering | None = None,
        embedding_store: EmbeddingStore | None = None,
    ) -> None:
        """Initialize the recommendation engine agent.

        Args:
            model: Optional pre-loaded NCF model.  If not provided a
                freshly-initialised model is created (suitable for tests).
            embedding_store: Optional shared embedding store.
        """
        super().__init__(name="recsys_engine")
        self._embedding_store = embedding_store or get_embedding_store()

        if model is not None:
            self._model = model
        else:
            logger.info(
                "Initializing NCF model (users=%d, items=%d, dim=%d)",
                _NUM_USERS, _NUM_ITEMS, _EMBEDDING_DIM,
            )
            self._model = NeuralCollaborativeFiltering(
                num_users=_NUM_USERS,
                num_items=_NUM_ITEMS,
                embedding_dim=_EMBEDDING_DIM,
            )
            self._model.eval()

        # Wire the embedding store to the model so agents share the same vectors.
        if not self._embedding_store.has_model:
            self._embedding_store.load_from_model(self._model)

    async def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Score candidate items and return a ranked list.

        Args:
            payload: Must conform to :class:`RecsysEngineInput` with
                ``user_profile`` (:class:`UserProfileOutput`),
                ``content_features`` (:class:`ContentAnalyzerOutput`),
                and ``top_k`` (int).

        Returns:
            A dict matching :class:`RecsysEngineOutput` with ``ranked_items``
            sorted by descending predicted score.
        """
        input_data = RecsysEngineInput(**payload)
        user_profile = input_data.user_profile
        content_features = input_data.content_features
        top_k = input_data.top_k

        logger.info(
            "Scoring %d candidates for user %d (top_k=%d)",
            len(content_features.items),
            user_profile.user_id,
            top_k,
        )

        if not content_features.items:
            logger.warning("No candidate items to score for user %d", user_profile.user_id)
            return RecsysEngineOutput(ranked_items=[]).model_dump()

        scored = self._score_items(user_profile, content_features)

        # Sort by score descending, take top_k
        scored.sort(key=lambda s: s.score, reverse=True)
        ranked = scored[:top_k]

        logger.info(
            "Top recommendation for user %d: item %d (score=%.4f)",
            user_profile.user_id,
            ranked[0].item_id if ranked else -1,
            ranked[0].score if ranked else 0.0,
        )

        output = RecsysEngineOutput(ranked_items=ranked)
        return output.model_dump()

    def _score_items(
        self,
        user_profile: UserProfileOutput,
        content_features: ContentAnalyzerOutput,
    ) -> list[ScoredItem]:
        """Run NCF inference on all candidate items.

        Uses :meth:`NeuralCollaborativeFiltering.predict_batch` for
        efficient batched scoring without gradients.

        Args:
            user_profile: The profiled user with embedding and ratings.
            content_features: Candidate items with embeddings.

        Returns:
            A list of :class:`ScoredItem` with model-predicted scores.
        """
        items = content_features.items

        # Clamp IDs to model vocabulary range.
        user_id = min(user_profile.user_id, self._model.num_users - 1)
        item_ids = [min(item.item_id, self._model.num_items - 1) for item in items]

        user_tensor = torch.full((len(items),), user_id, dtype=torch.long)
        item_tensor = torch.tensor(item_ids, dtype=torch.long)

        scores = self._model.predict_batch(user_tensor, item_tensor)

        scored_items: list[ScoredItem] = []
        for item, score in zip(items, scores.tolist()):
            scored_items.append(
                ScoredItem(
                    item_id=item.item_id,
                    title=item.title,
                    score=round(score, 4),
                    genres=item.genres,
                )
            )

        logger.debug(
            "Scored %d items, score range [%.4f, %.4f]",
            len(scored_items),
            min(s.score for s in scored_items),
            max(s.score for s in scored_items),
        )

        return scored_items
