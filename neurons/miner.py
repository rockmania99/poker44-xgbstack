"""poker44-xgbstack miner (v29a line): tree-stack detector, raw-probability serving.

Drop-in miner for Poker44 subnet 126. Requires the Poker44-subnet package
importable (pip install -e . in the subnet repo) for BaseMinerNeuron and
DetectionSynapse. Loads the local conformal artifact and returns one risk score
per chunk, calibrated so chunk-level FPR stays under the validator's 10% cliff.

Run:
  POKER44_BUMP_MODEL=$(pwd)/models/bump_model.joblib \
  python neurons/miner.py --netuid 126 --wallet.name <c> --wallet.hotkey <h> \
    --subtensor.network finney --axon.port 8091 \
    --blacklist.allowed_validator_hotkeys <vali_hotkey...>
"""
# NOTE: do NOT add `from __future__ import annotations` here — bittensor's
# axon.attach() introspects forward()'s annotations at runtime with issubclass(),
# which fails if they are stringized. Keep annotations as real class objects.
import os, time, hashlib, subprocess
from pathlib import Path
from typing import Tuple

import bittensor as bt
import joblib


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def _norm_repo_url(url: str) -> str:
    u = (url or "").strip()
    if u.startswith("git@") and ":" in u:
        host, path = u[4:].split(":", 1)
        u = f"https://{host}/{path}"
    return u[:-4] if u.endswith(".git") else u

from poker44.base.miner import BaseMinerNeuron
from poker44.validator.synapse import DetectionSynapse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from poker44_bump.live_capture import capture as _live_capture  # noqa: E402
except Exception:
    def _live_capture(*a, **k):  # capture optional; never break import
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        root = Path(__file__).resolve().parents[1]
        self.model_path = Path(os.getenv("POKER44_BUMP_MODEL", str(root / "models" / "bump_model.joblib")))
        self.model = joblib.load(self.model_path)
        md = self.model.metadata
        # self-derive repo identity from the local git checkout (accurate to deployed code)
        git_url = _norm_repo_url(_git(root, "config", "--get", "remote.origin.url"))
        git_commit = _git(root, "rev-parse", "HEAD")
        repo_url = git_url or md.get("repo_url", "")
        repo_commit = git_commit or md.get("repo_commit", "")
        self.model_manifest = {
            "schema_version": "1",
            "open_source": True,
            "repo_url": repo_url,
            "repo_commit": repo_commit,
            "model_name": md.get("model_name", "poker44-bump-robust"),
            "model_version": md.get("model_version", "bump-conformal-v1"),
            "framework": md.get("framework", "tree-ensemble+conformal"),
            "license": "MIT",
            "inference_mode": "local-joblib",
            "training_data_statement": md.get("training_data_statement") or (
                f"Trained on {md.get('benchmark_rows','?')} released benchmark chunks "
                f"(groundTruth labels) across dates {md.get('train_source_dates',['?'])[0]}.."
                f"{md.get('train_source_dates',['?'])[-1]}. Cliff-robust conformal head "
                f"(T={md.get('conformal_threshold'):.4f})."
            ),
            "training_data_sources": md.get("training_data_sources") or ["released_training_benchmark"],
            # attestation is MODEL-DRIVEN: benchmark-only models keep the default; models that
            # use the live query distribution must declare it truthfully via metadata.
            "private_data_attestation": md.get("data_attestation") or "No validator-private data used; released benchmark labels only.",
            "data_attestation": md.get("data_attestation") or "No validator-private data used; released benchmark labels only.",
            "implementation_files": sorted(
                str(p.relative_to(root)) for pat in ("neurons/miner.py", "poker44_bump/*.py", "poker44_ml/*.py")
                for p in root.glob(pat) if p.name != "__init__.py"
            ),
            "implementation_sha256": _sha256(Path(__file__).resolve()),
            "artifact_sha256": _sha256(self.model_path) if self.model_path.is_file() else "",
            "notes": f"head={md.get('serving_head', 'raw meta probability')}, oof_ap={md.get('oof_ap')}, combiner={md.get('ensemble_combiner')}",
        }
        bt.logging.info(
            f"\U0001f916 Poker44 bump miner | T={self.model.threshold:.4f} "
            f"feats={len(getattr(self.model, 'feature_names', []) or [])} oof_ap={md.get('oof_ap')}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = [list(c or []) for c in (synapse.chunks or [])]
        t0 = time.perf_counter()
        try:
            scores = self.model.predict_chunk_scores(chunks)
        except Exception as err:
            bt.logging.warning(f"scoring failed ({err}); returning 0.5 for all chunks")
            scores = [0.5] * len(chunks)
        scores = [max(0.0, min(1.0, float(s))) for s in scores]
        try:
            _hk = str(getattr(self.config.wallet, "hotkey", "self"))
            _vk = str(getattr(getattr(synapse, "dendrite", None), "hotkey", "") or "")
            _live_capture(chunks, scores, _hk, _vk)
        except Exception:
            pass
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        dt = (time.perf_counter() - t0) * 1000.0
        bt.logging.info(
            f"scored {len(chunks)} chunks | {dt:.1f}ms ({dt/max(len(chunks),1):.2f}ms/chunk) "
            f"bots={sum(synapse.predictions)} range=[{min(scores) if scores else 0:.3f},"
            f"{max(scores) if scores else 0:.3f}]"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 bump miner running...")
        while True:
            bt.logging.info(f"UID {miner.uid} | Incentive {miner.metagraph.I[miner.uid]}")
            time.sleep(300)
