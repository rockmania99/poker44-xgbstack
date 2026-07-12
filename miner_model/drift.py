"""Feature-drift report — which of your features to distrust on the live feed.

Adopts the pd-coast (UID 54) idea: the eval/live data is distributionally different
from training. You can't measure accuracy there (no labels), but you *can* measure
how far each feature's distribution has moved. Features that drift most (high PSI/KS)
are the ones most likely to mislead the model live — candidates to downweight or drop.

PSI guide:  <0.10 stable · 0.10-0.25 moderate shift · >0.25 major shift.

Usage:
  python -m miner_model.drift                    # train days vs newest held-out day
  python -m miner_model.drift --eval-day 2026-07-09
  python -m miner_model.drift --top 30
  python -m miner_model.drift --min-entries 5    # full days only (drop pruned old days)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from miner_model.data import available_dates, load_batches  # noqa: E402
from miner_model.features import batch_features  # noqa: E402


def _entry_counts(data_dir: str) -> dict:
    """Raw chunk-entries per day — cleanly separates full days (~18-21) from pruned (~2)."""
    counts = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "benchmark_*.json"))):
        day = os.path.basename(path)[len("benchmark_"):-len(".json")]
        try:
            counts[day] = len(json.load(open(path)))
        except Exception:
            counts[day] = 0
    return counts


def _matrix(batches) -> pd.DataFrame:
    return pd.DataFrame([batch_features(b) for b in batches]).fillna(0.0)


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index over quantile bins of the training distribution."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.clip(np.histogram(expected, bins=edges)[0] / max(len(expected), 1), 1e-6, None)
    a = np.clip(np.histogram(actual, bins=edges)[0] / max(len(actual), 1), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max gap between empirical CDFs)."""
    exp_sorted, act_sorted = np.sort(expected), np.sort(actual)
    grid = np.concatenate([exp_sorted, act_sorted])
    cdf_e = np.searchsorted(exp_sorted, grid, side="right") / len(exp_sorted)
    cdf_a = np.searchsorted(act_sorted, grid, side="right") / len(act_sorted)
    return float(np.max(np.abs(cdf_e - cdf_a)))


def _perm_test(train: np.ndarray, live: np.ndarray, n_perm: int, seed: int = 0):
    """Permutation test for genuine drift.

    At ~150 chunks/day, PSI>0.25 fires on pure sampling noise (a random split of ONE
    day flags ~45 'MAJOR' features). So we compare the observed train-vs-live PSI to a
    null built by repeatedly re-splitting the POOLED chunks. Returns (observed_psi,
    null_p95, p_value). A feature is *really* drifting only when p_value is small.
    """
    obs = _psi(train, live)
    pool = np.concatenate([train, live])
    n_tr = len(train)
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(len(pool))
        null[i] = _psi(pool[perm[:n_tr]], pool[perm[n_tr:]])
    p_value = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, float(np.quantile(null, 0.95)), p_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data"))
    parser.add_argument("--eval-day", default=None, help="Day to treat as 'live' (default: newest).")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--min-entries", type=int, default=0,
                        help="Only use days with >= this many chunk-entries "
                             "(e.g. 5 keeps full days, drops pruned old days).")
    parser.add_argument("--permute", type=int, default=300,
                        help="Permutation-null size for noise-corrected drift "
                             "(0 = raw PSI only, no significance test).")
    args = parser.parse_args()

    dates = available_dates(args.data_dir)
    if args.min_entries > 0:
        counts = _entry_counts(args.data_dir)
        kept = [d for d in dates if counts.get(d, 0) >= args.min_entries]
        dropped = [d for d in dates if d not in kept]
        if dropped:
            print(f"[--min-entries {args.min_entries}] dropped {len(dropped)} pruned/sparse day(s): "
                  f"{dropped[0]}..{dropped[-1]}")
        dates = kept
    if len(dates) < 2:
        raise SystemExit("Need >=2 days of data to measure drift. Harvest more (or lower --min-entries).")
    eval_day = args.eval_day or dates[-1]
    train_days = [d for d in dates if d != eval_day]
    print(f"train days ({len(train_days)}): {train_days[0]}..{train_days[-1]}   vs   'live' day: {eval_day}\n")

    train = _matrix([h for h, _, _ in load_batches(args.data_dir, set(train_days))])
    live = _matrix([h for h, _, _ in load_batches(args.data_dir, {eval_day})])
    cols = sorted(set(train.columns) & set(live.columns))

    rows = []
    for col in cols:
        a, b = train[col].to_numpy(float), live[col].to_numpy(float)
        if args.permute > 0:
            psi, null_p95, p_value = _perm_test(a, b, args.permute)
        else:
            psi, null_p95, p_value = _psi(a, b), float("nan"), float("nan")
        rows.append((col, psi, _ks(a, b), float(a.mean()), float(b.mean()), null_p95, p_value))

    if args.permute > 0:
        # Rank by how far the observed PSI clears the noise floor (excess over null p95).
        rows.sort(key=lambda r: -(r[1] - r[5]))

        def flag(psi, null_p95, p):
            if p < 0.05 and psi > null_p95:
                return "REAL" if psi > 2 * null_p95 else "real"
            return "noise"
    else:
        rows.sort(key=lambda r: -r[1])

        def flag(psi, null_p95, p):
            return "MAJOR" if psi > 0.25 else ("moderate" if psi > 0.10 else "stable")

    print(f"{'feature':38} {'PSI':>7} {'null95':>7} {'p':>6} {'train_mean':>11} {'live_mean':>10}  drift")
    print("-" * 96)
    for name, psi, ks, tm, lm, null_p95, p in rows[:args.top]:
        print(f"{name:38} {psi:7.3f} {null_p95:7.3f} {p:6.3f} {tm:11.3f} {lm:10.3f}  {flag(psi, null_p95, p)}")

    if args.permute > 0:
        real = [r for r in rows if r[6] < 0.05 and r[1] > r[5]]
        strong = [r for r in real if r[1] > 2 * r[5]]
        print(f"\nSummary: {len(cols)} features | GENUINE drift (perm p<0.05, above noise): {len(real)}"
              f"  of which strong (>2x noise floor): {len(strong)} | rest is sampling noise at n~{len(live)}.")
        if real:
            print("Genuinely unstable on live (downweight/drop these first): "
                  + ", ".join(name for name, *_ in real[:12]))
        else:
            print("No feature clears the sampling-noise floor — no evidence of real drift on this day.")
        print("(Raw PSI>0.25 counts are meaningless here: a random split of ONE day flags ~45 'MAJOR' "
              "features. Trust the permutation p-value, not raw PSI.)")
    else:
        major = [r for r in rows if r[1] > 0.25]
        moderate = [r for r in rows if 0.10 < r[1] <= 0.25]
        print(f"\nSummary: {len(cols)} features | MAJOR shift (PSI>0.25): {len(major)} | "
              f"moderate (0.10-0.25): {len(moderate)} | stable: {len(cols) - len(major) - len(moderate)}")
        print("WARNING: raw PSI at small n is noise-dominated; re-run with --permute 300 for a real test.")


if __name__ == "__main__":
    main()
