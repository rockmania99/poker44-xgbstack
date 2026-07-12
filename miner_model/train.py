"""Train + grade the chunk model locally against the REAL validator reward().

Uses a temporal holdout (train on older days, test on the newest held-out day)
so the score reflects unseen future players — the honest generalization signal.

Usage:
  python -m miner_model.train                     # auto: newest day = test, rest = train
  python -m miner_model.train --test-date 2026-07-09
  python -m miner_model.train --data-dir data --out artifacts/poker44_model.joblib
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Make the repo's real scorer importable (pure numpy+sklearn, no bittensor).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from poker44.score.scoring import reward  # noqa: E402  (the exact validator reward)
from miner_model.data import available_dates, load_batches  # noqa: E402
from miner_model.model import Poker44ChunkModel  # noqa: E402


def _report(title: str, scores: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    r, m = reward(scores, y)
    bot = scores[y == 1]
    human = scores[y == 0]
    crossed = int((scores >= 0.5).sum())
    gate_ok = crossed > 0 and m["threshold_sanity_quality"] > 0
    print(f"\n=== {title} ({len(y)} chunks: {int(y.sum())} bot / {int((1 - y).sum())} human) ===")
    print(f"  REWARD           : {r:.4f}      (champion bar ~0.56-0.59)")
    print(f"  AP (0.35)        : {m['ap_score']:.4f}")
    print(f"  bot_recall@5%FPR : {m['bot_recall']:.4f}   fpr={m['fpr']:.4f}")
    print(f"  threshold_sanity : {m['threshold_sanity_quality']:.4f}   hard_fpr={m.get('hard_fpr', 0):.4f}")
    print(f"  scores >= 0.5    : {crossed}/{len(y)}   0.5-gate: {'PASS' if gate_ok else 'FAIL (reward=0!)'}")
    if len(bot) and len(human):
        print(f"  bot median={np.median(bot):.3f}  human median={np.median(human):.3f}  "
              f"separation={np.median(bot) - np.median(human):+.3f}")
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data"))
    parser.add_argument("--test-date", default=None, help="Held-out day (default: newest available).")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-sequence", action="store_true", help="Trees only; skip the neural Set Transformer.")
    parser.add_argument("--seq-epochs", type=int, default=8, help="Training epochs for the Set Transformer.")
    parser.add_argument("--human-weight", type=float, default=1.0,
                        help="Fit weight on human chunks (>1 protects humans -> lower FPR).")
    parser.add_argument("--out", default=os.path.join(_PROJECT_ROOT, "artifacts", "poker44_model.joblib"))
    args = parser.parse_args()

    dates = available_dates(args.data_dir)
    if not dates:
        raise SystemExit(f"No benchmark_*.json found in {args.data_dir}. Run data/harvest_benchmark.py first.")
    test_date = args.test_date or dates[-1]
    train_dates = [d for d in dates if d != test_date]
    print(f"days available : {dates}")
    print(f"train days     : {train_dates}")
    print(f"holdout day    : {test_date}")

    train = load_batches(args.data_dir, dates=set(train_dates))
    test = load_batches(args.data_dir, dates={test_date})
    if not train or not test:
        raise SystemExit("Not enough data for a temporal split — harvest more days.")

    model = Poker44ChunkModel(
        use_sequence=not args.no_sequence, seq_epochs=args.seq_epochs,
        human_weight=args.human_weight,
    ).fit([h for h, _, _ in train], [y for _, y, _ in train], n_splits=args.folds)

    # In-training generalization estimate (out-of-fold, calibrated).
    _report("CROSS-VALIDATION (out-of-fold on train days)", model.calibrated_oof(), model.oof_labels_)

    # Honest metric: an entire future day the model never saw.
    test_scores = model.predict_scores([h for h, _, _ in test])
    holdout_reward = _report(f"TEMPORAL HOLDOUT (unseen day {test_date})", test_scores, [y for _, y, _ in test])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"\nsaved model -> {args.out}")
    print(f"HEADLINE: holdout reward = {holdout_reward:.4f} on unseen day {test_date}")


if __name__ == "__main__":
    main()
