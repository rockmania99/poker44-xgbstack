"""Poker44 miner model — a from-scratch implementation of the strategy:

behavioral feature engineering (with conditional-entropy, bet-size quantization,
and self-similarity signals the champion lacks) -> a CPU tree ensemble
(LightGBM + HistGradientBoosting + ExtraTrees) -> isotonic calibration to the
0.5 threshold -> graded with the *real* validator reward() and a temporal
holdout so the score reflects unseen future data.
"""

from .model import Poker44ChunkModel

__all__ = ["Poker44ChunkModel"]
