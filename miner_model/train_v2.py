"""V2 ablation trainer: corpus-human augmentation, drift-pruned features, size-mix.

Each flag is one of the three evidence-backed upgrades the V1 recipe lacks:
  --corpus     add real-human corpus chunks (anti-inversion; the 07-04 king edge)
  --prune      drop features that GENUINELY drift (perm-test p<.05 vs newest day)
               AND carry no bot/human signal (|single-feature AUC - .5| < .05)
  --size-mix   add large (80-105h) training chunks for BOTH classes (live chunks
               are pinned to 100 hands; benchmark is 30-40)

Evaluation (all on the newest full day, never trained on):
  1. pooled reward (comparable to train.py's headline)
  2. PER-WINDOW replay with the serving batch-relative head applied — the
     live-shaped metric (one chunk-entry = one graded window)
  3. inversion check: held-out corpus humans must rank BELOW the day's bots

Usage:
  python -m miner_model.train_v2 --tag C1 --corpus --out artifacts/v2_C1.joblib
  python -m miner_model.train_v2 --tag C2 --corpus --prune --out artifacts/v2_C2.joblib
  python -m miner_model.train_v2 --tag C3 --corpus --prune --size-mix 0.25 --out artifacts/v2_C3.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from poker44.score.scoring import reward  # noqa: E402
from miner_model.data import available_dates, load_batches  # noqa: E402
from miner_model.model import Poker44ChunkModel  # noqa: E402
from miner_model.corpus import load_corpus_chunks, bootstrap_to_size  # noqa: E402
from miner_model.drift import _matrix as drift_matrix, _perm_test  # noqa: E402

MIN_ENTRIES_FULL_DAY = 5


def full_days(data_dir: str, min_entries: int = MIN_ENTRIES_FULL_DAY) -> list[str]:
    days = []
    for d in available_dates(data_dir):
        with open(os.path.join(data_dir, f"benchmark_{d}.json")) as fh:
            if len(json.load(fh)) >= min_entries:
                days.append(d)
    return days


def batch_head(scores, frac=0.15):
    """Mirror of the serving batch-relative head (neurons/eros0*_miner.py)."""
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    k = max(1, int(round(frac * n)))
    out = [0.0] * n
    for rank, idx in enumerate(order):
        if rank < k:
            out[idx] = 0.509 - 0.008 * (rank / max(1, k - 1)) if k > 1 else 0.509
        else:
            r, m = rank - k, n - k
            out[idx] = 0.49 - 0.47 * (r / max(1, m - 1)) if m > 1 else 0.49
    return out


def pruned_features(train_batches, newest_batches, train_labels, n_perm=200) -> list[str]:
    """Features that genuinely drift toward the newest day AND separate nothing."""
    tr = drift_matrix(train_batches)
    nw = drift_matrix(newest_batches).reindex(columns=tr.columns, fill_value=0.0)
    y = np.asarray(train_labels, dtype=int)
    drop = []
    for col in tr.columns:
        a, b = tr[col].to_numpy(float), nw[col].to_numpy(float)
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            continue
        _, _, p = _perm_test(a, b, n_perm)
        if p >= 0.05:
            continue
        # single-feature separation on train (rank AUC via mean rank of bots)
        order = np.argsort(np.argsort(a))
        auc = order[y == 1].mean() / max(len(a) - 1, 1) - order[y == 0].mean() / max(len(a) - 1, 1)
        if abs(auc) < 0.05:
            drop.append(col)
    return drop


def report(title, scores, labels):
    y = np.asarray(labels, dtype=int)
    r, m = reward(np.asarray(scores, float), y)
    print(f"  {title}: reward={r:.4f} AP={m['ap_score']:.4f} recall@5%FPR={m['bot_recall']:.4f} "
          f"sanity={m['human_safety_penalty']:.2f} hard_fpr={m['hard_fpr']:.3f}", flush=True)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--corpus-small", type=int, default=600)
    ap.add_argument("--corpus-large", type=int, default=150)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--drop-names", default="", help="comma-separated feature names to hard-drop "
                    "(e.g. hand_count: has benchmark signal but is guaranteed off-distribution at live "
                    "100-hand chunks, so the drift screen against benchmark days cannot catch it)")
    ap.add_argument("--size-mix", type=float, default=0.0, help="fraction of bench chunks to bootstrap to 80-105h")
    ap.add_argument("--drop-file", default="", help="file with newline-separated feature names to drop "
                    "(e.g. artifacts/stable_drop.txt from screen_stability)")
    ap.add_argument("--recency", default="", help="K,M — weight the newest K bench days xM (e.g. 2,3)")
    ap.add_argument("--min-entries", type=int, default=MIN_ENTRIES_FULL_DAY,
                    help="include benchmark days with at least this many entries (1 = the "
                    "2-entry hoard days from 2026-05-26..07-05, generator v1.12)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--data-dir", default=os.path.join(_PROJECT_ROOT, "data"))
    ap.add_argument("--test-date", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    days = full_days(args.data_dir, min_entries=args.min_entries)
    test_date = args.test_date or full_days(args.data_dir)[-1]  # holdout stays a FULL day
    train_days = [d for d in days if d != test_date]
    print(f"[{args.tag}] train days {train_days} | holdout {test_date}", flush=True)

    bench = load_batches(args.data_dir, dates=set(train_days))
    test = load_batches(args.data_dir, dates={test_date})
    chunks = [h for h, _, _ in bench]
    labels = [y for _, y, _ in bench]
    print(f"[{args.tag}] bench chunks {len(chunks)}", flush=True)

    corpus_holdout = []
    if args.corpus:
        c_small, c_large, corpus_holdout = load_corpus_chunks(
            max_small=args.corpus_small, max_large=args.corpus_large, seed=args.seed)
        chunks += c_small
        labels += [0] * len(c_small)
        if args.size_mix > 0:
            chunks += c_large
            labels += [0] * len(c_large)
            # give bots (and some bench humans) large sizes too -> no size shortcut
            bench_idx = [i for i in range(len(bench))]
            rng.shuffle(bench_idx)
            n_boot = int(args.size_mix * len(bench))
            boots, blabels = [], []
            for i in bench_idx[:n_boot]:
                h, y, _ = bench[i]
                boots.append(bootstrap_to_size(h, rng=rng))
                blabels.append(y)
            chunks += boots
            labels += blabels
            print(f"[{args.tag}] size-mix: +{len(c_large)} real-large humans, "
                  f"+{len(boots)} bootstrapped bench ({sum(blabels)} bot/{len(blabels)-sum(blabels)} human)", flush=True)
        print(f"[{args.tag}] +corpus: total {len(chunks)} chunks "
              f"({sum(labels)} bot / {len(labels)-sum(labels)} human)", flush=True)

    drop = [n for n in args.drop_names.split(",") if n.strip()]
    if args.drop_file:
        drop += [ln.strip() for ln in open(args.drop_file) if ln.strip()]
    if args.prune:
        drop += pruned_features([h for h, _, _ in bench], [h for h, _, _ in test],
                                [y for _, y, _ in bench])
        print(f"[{args.tag}] pruning {len(drop)} drift-fragile/no-signal features", flush=True)
    if drop:
        print(f"[{args.tag}] dropping: {drop[:6]}{'...' if len(drop) > 6 else ''}", flush=True)

    weights = None
    if args.recency:
        k, m = (int(x) for x in args.recency.split(","))
        recent = set(train_days[-k:])
        # bench chunks carry their source date; corpus/augmented chunks stay at 1.0
        weights = [m if (i < len(bench) and bench[i][2] in recent) else 1.0
                   for i in range(len(chunks))]
        print(f"[{args.tag}] recency x{m} on {sorted(recent)} "
              f"({sum(1 for w in weights if w > 1)} chunks)", flush=True)

    model = Poker44ChunkModel(seed=args.seed, use_sequence=False, drop_features=drop).fit(
        chunks, labels, n_splits=args.folds, weights=weights)

    # 1. pooled holdout
    test_chunks = [h for h, _, _ in test]
    test_labels = [y for _, y, _ in test]
    scores = model.predict_scores(test_chunks)
    report(f"POOLED holdout {test_date}", scores, test_labels)

    # 2. per-window replay with the serving head (the live-shaped metric)
    entries = json.load(open(os.path.join(args.data_dir, f"benchmark_{test_date}.json")))
    wr = []
    for e in entries:
        wl = [1 if l == "bot" else 0 for l in e["groundTruthLabels"]]
        ws = model.predict_scores(e["chunks"])
        v, _ = reward(np.asarray(batch_head(list(ws)), float), np.asarray(wl))
        wr.append(v)
    print(f"  PER-WINDOW(head): mean={np.mean(wr):.4f} median={np.median(wr):.4f} "
          f"min={min(wr):.3f} zeros={sum(1 for v in wr if v == 0)}/{len(wr)}", flush=True)

    # 3. inversion check: held-out corpus humans vs the day's bots
    if corpus_holdout:
        bot_chunks = [c for c, y in zip(test_chunks, test_labels) if y == 1]
        s_h = model.predict_scores(corpus_holdout)
        s_b = model.predict_scores(bot_chunks)
        inv_labels = np.array([0] * len(s_h) + [1] * len(s_b))
        inv_scores = np.concatenate([s_h, s_b])
        order = np.argsort(np.argsort(inv_scores))
        auc = (order[inv_labels == 1].mean() - order[inv_labels == 0].mean()) / max(len(inv_scores) - 1, 1) + 0.5
        print(f"  INVERSION: corpus-human median={np.median(s_h):.3f} bot median={np.median(s_b):.3f} "
              f"AUC(bot>human)={auc:.3f} {'OK' if auc > 0.6 else 'INVERTED/WEAK'}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"[{args.tag}] saved -> {args.out} (features={len(model.columns)}, "
          f"op={model.op_threshold})", flush=True)


if __name__ == "__main__":
    main()
