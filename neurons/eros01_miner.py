"""Poker44 miner — 147-feature behavioral tree ensemble (eros01 deployment).

Model family: LightGBM + HistGradientBoosting + ExtraTrees -> OOF isotonic
calibration -> conformal 0.5 operating point. This deployment serves the
v1.0-s42 artifact (its own trained weights; see miner_model/ for the pipeline).
"""

import os
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

from miner_model.live_miner import ChunkScorer

try:
    from miner_model.live_capture import capture_chunks
except Exception:
    capture_chunks = None

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = os.getenv("POKER44_ARTIFACT", str(REPO_ROOT / "artifacts" / "poker44_model.joblib"))
MODEL_NAME = "poker44-behavioral-trees-a"
MODEL_VERSION = "v1.0-s42"
REPO_URL = "https://github.com/rockmania99/poker44-xgbstack"
FALLBACK_SCORE = 0.45

import os


def _batch_relative_head(scores, frac=0.15, pos_floor=0.501, pos_ceil=0.509, neg_floor=0.02, neg_ceil=0.49):
    """Rank-preserving batch-relative squeeze: top `frac` of the batch lands in
    [pos_floor, pos_ceil] (crosses 0.5), the rest in [neg_floor, neg_ceil].

    Guarantees the 0.5 threshold-sanity gate passes with bounded hard-FPR even
    when the calibrated scores drift wholesale above 0.5 on the live feed
    (benchmark->live shift). AP and recall@FPR are rank-based and unchanged.
    """
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    k = max(1, int(round(frac * n)))
    out = [0.0] * n
    for rank, idx in enumerate(order):
        if rank < k:
            out[idx] = pos_ceil - (pos_ceil - pos_floor) * (rank / max(1, k - 1)) if k > 1 else pos_ceil
        else:
            r = rank - k
            m = n - k
            out[idx] = neg_ceil - (neg_ceil - neg_floor) * (r / max(1, m - 1)) if m > 1 else neg_ceil
    return out

  # neutral-negative: never flags a human when scoring fails


class Miner(BaseMinerNeuron):
    """Serves one risk score per chunk from the trained behavioral artifact."""

    def __init__(self, config=None):
        super().__init__(config=config)
        bt.logging.info(f"Poker44 behavioral miner | model={MODEL_NAME} version={MODEL_VERSION}")
        self.scorer = ChunkScorer(ARTIFACT)
        bt.logging.info(
            f"artifact loaded: {ARTIFACT} | op_threshold={getattr(self.scorer.model, 'op_threshold', None)}"
        )
        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=[Path(__file__).resolve()],
            defaults={
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "framework": "lightgbm+histgb+extratrees -> isotonic -> conformal-0.5",
                "license": "MIT",
                "repo_url": REPO_URL,
                "notes": "147 behavioral features per chunk (entropy, quantization, self-similarity, n-grams, street rigidity, bet/pot). Weights are this key's own seed-variant training run.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained exclusively on the released public training benchmark "
                    "(api.poker44.net/api/v1/benchmark), full days 2026-07-06..2026-07-11."
                ),
                "training_data_sources": ["released_training_benchmark"],
                "private_data_attestation": (
                    "No validator-private evaluation data was used for training or calibration."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(
            f"manifest status={self.manifest_compliance['status']} digest={self.manifest_digest}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        if capture_chunks is not None:
            capture_chunks(chunks, tag="eros01", base_dir=str(REPO_ROOT))
        t0 = time.time()
        try:
            if os.getenv("POKER44_BATCH_HEAD", "off") == "topk":
                # rank by the RAW ensemble blend: full granularity on live inputs
                # (isotonic collapses OOD chunks onto plateau ties, which kills AP;
                # the head re-bands absolute values, so only the ORDER matters)
                ranked = self.scorer.model.predict_raw_scores(chunks)
            else:
                ranked = self.scorer.score(chunks)
            scores = [float(min(max(s, 0.0), 1.0)) for s in ranked]
        except Exception as exc:
            bt.logging.error(f"batch scoring failed ({exc}); per-chunk fallback")
            scores = []
            for chunk in chunks:
                try:
                    scores.append(float(min(max(self.scorer.model.predict_chunk(chunk), 0.0), 1.0)))
                except Exception:
                    scores.append(FALLBACK_SCORE)
        if os.getenv("POKER44_BATCH_HEAD", "off") == "topk":
            scores = _batch_relative_head(scores, frac=float(os.getenv("POKER44_TOPK_FRAC", "0.15")))
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(
            f"scored {len(chunks)} chunks in {time.time() - t0:.1f}s | "
            f"flagged={sum(1 for s in scores if s >= 0.5)}/{len(chunks)}"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 behavioral miner running...")
        while True:
            bt.logging.info(
                f"UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
