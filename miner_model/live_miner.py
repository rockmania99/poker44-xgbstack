"""Deployment path: load the trained artifact and score chunks for the live miner.

When you go live (needs the bittensor venv + a registered hotkey), wire this into
neurons/miner.py: build one ChunkScorer in __init__, then in forward() do
    synapse.risk_scores = scorer.score(synapse.chunks)
    synapse.predictions = [s >= 0.5 for s in synapse.risk_scores]
"""

from __future__ import annotations

import os
from typing import List

from .model import Poker44ChunkModel

_DEFAULT_ARTIFACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "poker44_model.joblib",
)


class ChunkScorer:
    """Loads the trained model once and scores a list of chunks -> risk scores."""

    def __init__(self, artifact_path: str = _DEFAULT_ARTIFACT):
        self.model = Poker44ChunkModel.load(artifact_path)

    def score(self, chunks: List[list]) -> List[float]:
        # One score per chunk, in order — exactly what the validator requires.
        return [self.model.predict_chunk(chunk) for chunk in (chunks or [])]
