"""v29 additions: weighted-mean ensemble (king-#1-style serve-time combination).

The r4-era review of the two leader lines showed the #1 model combines member
probabilities by a plain weighted MEAN (weights in the artifact) and serves the
result raw (no fixed-fraction squeeze), while the #2 line still squeezes — and
the #1 line degraded less when v2.2 evaluation landed. WeightedMeanEnsemble is
our independent implementation of that combination idea: members are our own
V23-style tree stacks, weights fit on grouped-OOF AP. No code shared with any
other miner's repository.
"""
from __future__ import annotations
import os
from typing import List, Sequence

import numpy as np

from poker44_bump.model_v5 import _topk_squeeze


class WeightedMeanEnsemble:
    """OOF-AP-weighted mean of per-member probabilities; raw-first serving."""

    def __init__(self, estimators, feature_names, weights, topk_cfg=None, metadata=None):
        self.estimators = list(estimators)
        self.feature_names = list(feature_names)
        w = np.asarray(weights, dtype=float)
        w = np.clip(w, 0.0, None)
        self.weights = (w / w.sum()) if w.sum() > 0 else np.full(len(self.estimators), 1.0 / len(self.estimators))
        self.topk_cfg = dict(topk_cfg or {"positive_fraction": 0.15})
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("model_name", "poker44-wmean")
        self.metadata.setdefault("ensemble_combiner", "oof-ap-weighted mean")
        self.metadata["topk_cfg"] = self.topk_cfg
        self.threshold = 0.5
        self.head_mode = "raw"
        self.subsample = False

    def _rows(self, chunks):
        from poker44_ml.features import chunk_features as base_cf
        rows = []
        for c in chunks:
            c = list(c or [])
            bf = base_cf(c) if c else {"hand_count": 0.0}
            bf["hand_count"] = float(len(c))
            rows.append([float(bf.get(n, 0.0)) for n in self.feature_names])
        return np.asarray(rows, dtype=np.float64)

    def predict_raw(self, chunks) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        X = self._rows(chunks)
        cols = np.stack([np.clip(e.predict_proba(X)[:, 1], 0, 1) for e in self.estimators], axis=1)
        return np.clip(cols @ self.weights, 0.0, 1.0)

    def predict_chunk_scores(self, chunks) -> List[float]:
        raw = self.predict_raw(chunks)
        if os.getenv("POKER44_HEAD", "raw") == "raw":
            return [float(x) for x in raw]
        frac = float(os.getenv("POKER44_TOPK_FRAC", self.topk_cfg.get("positive_fraction", 0.15)))
        return _topk_squeeze(raw, frac,
                             float(self.topk_cfg.get("positive_floor", 0.501)),
                             float(self.topk_cfg.get("positive_ceiling", 0.509)),
                             float(self.topk_cfg.get("negative_ceiling", 0.49)))

    def score_chunk(self, chunk) -> float:
        return self.predict_chunk_scores([chunk])[0]
