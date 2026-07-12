"""CPU tree ensemble + isotonic calibration for chunk-level bot detection.

Base learners: LightGBM + HistGradientBoosting + ExtraTrees (all CPU, all use
every core). Their probabilities are averaged, then an isotonic calibrator fit
on out-of-fold scores maps the blend to a probability centered on the 0.5
threshold (so bots land above 0.5, humans below — satisfying the reward gate).
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from .features import batch_features

# The exact validator reward() (pure numpy+sklearn) — used to tune the operating point.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from poker44.score.scoring import reward  # noqa: E402


def _build_models(seed: int = 42) -> "OrderedDict":
    return OrderedDict(
        lgbm=lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1,
        ),
        hgb=HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=False, random_state=seed,
        ),
        et=ExtraTreesClassifier(
            n_estimators=400, max_depth=None, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
    )


class Poker44ChunkModel:
    """Trainable, picklable chunk-level bot detector. `predict_chunk(hands) -> float`."""

    def __init__(self, seed: int = 42, use_sequence: bool = True, seq_epochs: int = 8,
                 human_weight: float = 1.0, drop_features=None):
        self.seed = seed
        self.use_sequence = use_sequence
        self.seq_kwargs = {"epochs": seq_epochs, "seed": seed}
        # >1.0 makes each human cost more to misclassify -> pushes humans down, hardening
        # the low-FPR region the reward grades on (recall@<=5%FPR). Classes are 50/50 so
        # this is pure asymmetry, not an imbalance fix.
        self.human_weight = float(human_weight)
        # Names to exclude from the feature matrix (drift-fragile families). Applied at
        # fit_columns time, so predict-time reindexing stays consistent automatically.
        self.drop_features = set(drop_features or [])
        self.models = None
        self.seq_model = None
        self.calibrator: IsotonicRegression | None = None
        self.op_threshold: float | None = None  # conformal operating point (maps to 0.5)
        self.columns: List[str] | None = None
        self.oof_scores_: np.ndarray | None = None
        self.oof_labels_: np.ndarray | None = None

    def _matrix(self, batches: List[list], *, fit_columns: bool = False) -> np.ndarray:
        frame = pd.DataFrame([batch_features(b) for b in batches])
        if fit_columns or self.columns is None:
            drop = getattr(self, "drop_features", set()) or set()
            self.columns = sorted(c for c in frame.columns if c not in drop)
        frame = frame.reindex(columns=self.columns, fill_value=0.0).fillna(0.0)
        return frame.to_numpy(dtype=float)

    def _sample_weight(self, y: np.ndarray) -> np.ndarray:
        """Per-chunk fit weight: humans (label 0) weighted by `human_weight`, bots at 1.0."""
        return np.where(np.asarray(y) == 0, self.human_weight, 1.0).astype(float)

    def fit(self, batches: List[list], labels, n_splits: int = 5, verbose: bool = True,
            weights=None) -> "Poker44ChunkModel":
        y = np.asarray(labels, dtype=int)
        batches = list(batches)
        X = self._matrix(batches, fit_columns=True)
        w = self._sample_weight(y)
        if weights is not None:
            w = w * np.asarray(weights, dtype=float)  # e.g. recency weighting by source date

        # Out-of-fold blend for an honest calibrator fit.
        n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros(len(y), dtype=float)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            preds = []
            for model in _build_models(self.seed).values():
                model.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
                preds.append(model.predict_proba(X[val_idx])[:, 1])
            if self.use_sequence:
                seq = self._new_sequence_model().fit([batches[i] for i in train_idx], y[train_idx])
                preds.append(seq.predict_proba([batches[i] for i in val_idx])[:, 1])
            oof[val_idx] = np.mean(preds, axis=0)
            if verbose:
                print(f"    fold {fold + 1}/{n_splits} done")

        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
        self.oof_scores_ = oof
        self.oof_labels_ = y

        # Conformal FPR control: shift the operating point so the 0.5 decision line
        # sits at a low-FPR threshold (picked to maximize validator reward on OOF).
        cal_oof = np.clip(self.calibrator.transform(oof), 0.0, 1.0)
        self.op_threshold = self._fit_operating_point(cal_oof, y)

        # Refit base learners on all data for deployment.
        self.models = _build_models(self.seed)
        for model in self.models.values():
            model.fit(X, y, sample_weight=w)
        self.seq_model = self._new_sequence_model().fit(batches, y) if self.use_sequence else None
        if verbose:
            op = "identity" if self.op_threshold is None else f"{self.op_threshold:.3f}->0.5"
            members = len(self.models) + (1 if self.seq_model is not None else 0)
            print(f"  trained {members}-model ensemble on {len(y)} chunks x {X.shape[1]} features "
                  f"({int(y.sum())} bot / {int((1 - y).sum())} human), {n_splits}-fold OOF calibration, "
                  f"operating point={op}, human_weight={self.human_weight:g}")
        return self

    def _new_sequence_model(self):
        from .sequence_model import SequenceModelClassifier  # lazy: torch only if used
        return SequenceModelClassifier(**self.seq_kwargs)

    @staticmethod
    def _remap(scores, threshold) -> np.ndarray:
        """Monotone piecewise-linear remap so `threshold` -> 0.5 (preserves ranking)."""
        s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        if threshold is None:
            return s
        t = float(min(max(threshold, 1e-3), 1.0 - 1e-3))
        return np.clip(np.where(s <= t, 0.5 * (s / t), 0.5 + 0.5 * (s - t) / (1.0 - t)), 0.0, 1.0)

    def _fit_operating_point(self, cal_oof: np.ndarray, y: np.ndarray):
        """Pick the FPR target (a human-score quantile) that maximizes reward on OOF."""
        y = np.asarray(y, dtype=int)
        human = cal_oof[y == 0]
        best_threshold, best_reward = None, -1.0
        for target_fpr in (None, 0.02, 0.03, 0.05, 0.08, 0.10):
            if target_fpr is None:
                threshold, remapped = None, cal_oof
            elif len(human) == 0:
                continue
            else:
                threshold = float(np.quantile(human, 1.0 - target_fpr))
                remapped = self._remap(cal_oof, threshold)
            candidate_reward, _ = reward(remapped, y)
            if candidate_reward > best_reward:
                best_reward, best_threshold = candidate_reward, threshold
        return best_threshold

    def _raw_blend(self, X: np.ndarray, batches: List[list]) -> np.ndarray:
        probs = [m.predict_proba(X)[:, 1] for m in self.models.values()]
        if self.seq_model is not None:
            probs.append(self.seq_model.predict_proba(batches)[:, 1])
        return np.mean(probs, axis=0)

    def predict_scores(self, batches: List[list]) -> np.ndarray:
        batches = list(batches)
        raw = self._raw_blend(self._matrix(batches), batches)
        cal = np.clip(self.calibrator.transform(raw), 0.0, 1.0)
        return self._remap(cal, self.op_threshold)

    def predict_chunk(self, hands: list) -> float:
        """One chunk (list of hands) -> bot-risk score in [0, 1]. Used by the live miner."""
        return float(self.predict_scores([hands])[0])

    def predict_raw_scores(self, batches: List[list]) -> np.ndarray:
        """Uncalibrated ensemble blend — full-granularity ORDERING for batch-relative
        serving heads. The isotonic calibrator collapses out-of-distribution (live)
        inputs onto a few plateau steps (measured: 7-13 unique values per 100 live
        chunks vs 99-100 for the raw blend), and tied scores destroy AP. When a head
        re-bands absolute values anyway, rank by this instead."""
        batches = list(batches)
        return np.clip(self._raw_blend(self._matrix(batches), batches), 0.0, 1.0)

    def calibrated_oof(self) -> np.ndarray:
        cal = np.clip(self.calibrator.transform(self.oof_scores_), 0.0, 1.0)
        return self._remap(cal, self.op_threshold)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Poker44ChunkModel":
        return joblib.load(path)
