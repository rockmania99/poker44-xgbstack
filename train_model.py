"""v29 line: 5 eros r5-test versions (build+improve pass over v28, informed by
the leader-repo review: raw serving everywhere; #1 combines members by weighted
mean; topk line degraded harder under v2.2 eval).
Improvements over v28: 5-fold grouped OOF (was 3), per-version families/seeds,
and a weighted-mean combiner variant. All train on every released date incl.
the v2.2 drop + released human corpus, size-mixed. Raw head.
Versions (VER=a|b|c|e):
  a  XGB/HistGB family + logistic meta, seed 311   -> eros-01/uid50
  b  LGBM family + logistic meta,      seed 307   -> eros-02/uid64
  c  6 members across 3 families, OOF-AP weighted-MEAN combiner (king-style),
     seed 331                                      -> eros-03/uid92
  e  GOSS/DART family + logistic meta, seed 353   -> eros-05/uid114
  (d = serve-time bag of a+b, assembled by build_v29d.py -> eros-04/uid162)
Run: cd /root/pocker44-miner && PYTHONPATH=. VER=a .venv/bin/python train_v29.py
"""
import sys, os, glob, gzip, json, time, warnings, numpy as np, joblib
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf
try:
    from poker44_bump.model_v23 import V23Model
except ImportError:  # vendored only where the served artifact needs it
    V23Model = None
try:
    from poker44_bump.model_v29 import WeightedMeanEnsemble
except ImportError:
    WeightedMeanEnsemble = None
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              HistGradientBoostingClassifier)
import lightgbm as lgb

VER = os.getenv("VER", "a").lower()
CFG = {
    "a": dict(seed=311, fam="xgb"),
    "b": dict(seed=307, fam="lgbm"),
    "c": dict(seed=331, fam="mixed6"),
    "e": dict(seed=353, fam="gossdart"),
}[VER]
SEED, FAM = CFG["seed"], CFG["fam"]
LARGE_FRAC, DOSE, AUG_PER_GROUP, FOLDS = 0.25, 700, 3, 5
RNG = np.random.default_rng(SEED)
CORPUS = "/root/Poker44-subnet/hands_generator/human_hands/poker_hands_combined.json.gz"
TAG = f"v29{VER}"
OUT = f"models/bump_model_v29{VER}.joblib"

def make_trees():
    if FAM == "xgb":
        from xgboost import XGBClassifier
        return [
            XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                          random_state=SEED, n_jobs=8, eval_metric="logloss"),
            XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.7,
                          colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=2.0, tree_method="hist",
                          random_state=SEED + 1, n_jobs=8, eval_metric="logloss"),
            HistGradientBoostingClassifier(max_depth=4, learning_rate=0.03, max_iter=600,
                                           l2_regularization=1.0, random_state=SEED + 2),
            ExtraTreesClassifier(n_estimators=800, min_samples_leaf=3, max_features=0.3,
                                 random_state=SEED + 3, n_jobs=8),
            RandomForestClassifier(n_estimators=600, min_samples_leaf=3, max_features=0.4,
                                   random_state=SEED + 4, n_jobs=8),
        ]
    if FAM == "gossdart":
        return [
            lgb.LGBMClassifier(boosting_type="goss", n_estimators=500, learning_rate=0.03,
                               num_leaves=31, min_child_samples=20, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=SEED, n_jobs=8, verbose=-1),
            lgb.LGBMClassifier(boosting_type="dart", n_estimators=400, learning_rate=0.05,
                               num_leaves=31, min_child_samples=20, subsample=0.8,
                               colsample_bytree=0.8, drop_rate=0.1, random_state=SEED + 1,
                               n_jobs=8, verbose=-1),
            HistGradientBoostingClassifier(max_depth=6, learning_rate=0.03, max_iter=700,
                                           l2_regularization=0.5, random_state=SEED + 2),
            ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=2, max_features=0.25,
                                 random_state=SEED + 3, n_jobs=8),
            RandomForestClassifier(n_estimators=600, min_samples_leaf=2, max_features=0.5,
                                   random_state=SEED + 4, n_jobs=8),
        ]
    if FAM == "mixed6":
        from xgboost import XGBClassifier
        return [
            lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=8, verbose=-1),
            lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=63, min_child_samples=30,
                               subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, random_state=SEED + 1, n_jobs=8, verbose=-1),
            XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                          random_state=SEED + 2, n_jobs=8, eval_metric="logloss"),
            XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.7,
                          colsample_bytree=0.6, reg_lambda=5.0, tree_method="hist",
                          random_state=SEED + 3, n_jobs=8, eval_metric="logloss"),
            HistGradientBoostingClassifier(max_depth=4, learning_rate=0.03, max_iter=600,
                                           l2_regularization=1.0, random_state=SEED + 4),
            ExtraTreesClassifier(n_estimators=800, min_samples_leaf=3, max_features=0.3,
                                 random_state=SEED + 5, n_jobs=8),
        ]
    return [  # lgbm family
        lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=8, verbose=-1),
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4, min_child_samples=40,
                           subsample=0.7, colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=2.0, random_state=SEED + 1, n_jobs=8, verbose=-1),
        lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=63, min_child_samples=30,
                           subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, random_state=SEED + 2, n_jobs=8, verbose=-1),
        ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5, random_state=SEED + 3, n_jobs=8),
        RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5, random_state=SEED + 4, n_jobs=8),
    ]

exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks, ys, groups = [], [], []
by_dl = {}
for e in exs:
    gid = f"bench:{e['source_date']}"; ch = e["chunk"]; lab = int(e["label"])
    chunks.append(ch); ys.append(lab); groups.append(gid)
    by_dl.setdefault((e["source_date"], lab), []).extend(ch)
    n = len(ch)
    for k in range(AUG_PER_GROUP):
        m = int(n * (0.7 + 0.1 * (k % 3)))
        idx = RNG.choice(n, size=max(20, m), replace=False)
        chunks.append([ch[i] for i in sorted(idx)]); ys.append(lab); groups.append(gid)
n_native = len(chunks)
n_large_bench = 0
for (d, lab), hands in sorted(by_dl.items()):
    hp = list(hands); RNG.shuffle(hp); pos = 0
    while pos + 80 <= len(hp):
        s = int(RNG.integers(80, 106)); s = min(s, len(hp) - pos)
        chunks.append(hp[pos:pos + s]); ys.append(lab); groups.append(f"bench:{d}")
        pos += s; n_large_bench += 1
with gzip.open(CORPUS, "rt") as f:
    corpus = json.load(f)
pos, n_cs, n_cl = 0, 0, 0
while pos + 30 <= len(corpus) and n_cs < DOSE:
    if RNG.random() < LARGE_FRAC and pos + 80 <= len(corpus):
        s = int(RNG.integers(80, 106)); n_cl += 1
    else:
        s = int(RNG.integers(30, 41)); n_cs += 1
    chunks.append(bd.miner_visible_chunk(corpus[pos:pos + s])); ys.append(0)
    groups.append(f"corpus:{pos // 1200}"); pos += s
y = np.array(ys, dtype=np.float32); groups = np.array(groups)
N = len(chunks)
print(f"[{TAG}] TOTAL {N} chunks | bots={int(y.sum())} | bench {n_native}+{n_large_bench}L | corpus {n_cs}+{n_cl}L", flush=True)

names = list(joblib.load("models/bump_model_v10.joblib").feature_names)
t0 = time.time()
X = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in (base_cf(c) for c in chunks)])
print(f"[{TAG}] featurized ({time.time()-t0:.0f}s)", flush=True)

gids = sorted(set(groups)); RNG.shuffle(gids)
folds = [set(gids[i::FOLDS]) for i in range(FOLDS)]
NM = len(make_trees())
OOF = np.full((N, NM), np.nan)
for fi, grp in enumerate(folds):
    te = [i for i in range(N) if groups[i] in grp]
    tr = [i for i in range(N) if groups[i] not in grp]
    for j, est in enumerate(make_trees()):
        est.fit(X[tr], y[tr]); OOF[te, j] = np.clip(est.predict_proba(X[te])[:, 1], 0, 1)
    print(f"  [{TAG}] fold {fi+1}/{FOLDS} ({time.time()-t0:.0f}s)", flush=True)

per_ap = [average_precision_score(y, OOF[:, j]) for j in range(NM)]
trees = make_trees()
for est in trees: est.fit(X, y)

STMT = ("Trained on RELEASED public data only: public benchmark releases (all sourceDates "
        "through 2026-07-06 incl. the v2.2 expanded release, groundTruth labels; within-group "
        "subset augmentation; per-date-per-label pooled 80-105-hand chunks for live-size "
        "coverage) + the canonical repo's released human hands corpus (32k real human hands, "
        "session blocks at both size ranges), sanitized via prepare_hand_for_miner. "
        "No validator-private data.")

if FAM == "mixed6":
    w = np.clip(np.asarray(per_ap) - 0.5, 0.01, None) ** 2   # emphasize stronger members
    model = WeightedMeanEnsemble(trees, names, w, metadata={
        "model_name": f"poker44-bump-{TAG}", "model_version": f"{TAG}-wmean-v22data-rawhead",
        "serving_head": "raw weighted-mean probability", "seed": SEED,
        "oof_ap": round(float(average_precision_score(y, np.clip(OOF, 0, 1) @ (w / w.sum()))), 4),
        "member_oof_ap": [round(float(a), 4) for a in per_ap],
        "n_train_chunks": N, "training_data_statement": STMT,
        "training_data_sources": ["released_training_benchmark", "subnet_repo_human_hands_corpus"],
        "data_attestation": "No validator-private data used; released benchmark + released repo corpus only."})
    ap_meta = model.metadata["oof_ap"]
else:
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(OOF, y)
    ap_meta = average_precision_score(y, meta.predict_proba(OOF)[:, 1])
    model = V23Model(trees, names, [], dict(d=64), meta, isotonic=None,
                     topk_cfg={"positive_fraction": 0.15},
                     metadata={"model_name": f"poker44-bump-{TAG}",
                               "model_version": f"{TAG}-{FAM}-v22data-rawhead",
                               "serving_head": "raw meta probability", "seed": SEED,
                               "oof_ap": round(float(ap_meta), 4), "n_train_chunks": N,
                               "training_data_statement": STMT,
                               "training_data_sources": ["released_training_benchmark", "subnet_repo_human_hands_corpus"],
                               "data_attestation": "No validator-private data used; released benchmark + released repo corpus only."})
joblib.dump(model, OUT)
print(f"[{TAG}] OOF AP META={float(ap_meta):.4f} | members={[round(a,3) for a in per_ap]} | saved {OUT}", flush=True)

# local test battery: latency + live-capture spread/uniq (raw head)
live = []
for f in sorted(glob.glob("live_capture/archive/capture_pes03*.jsonl*")):
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        for l in fh:
            try: live.append(json.loads(l)["chunk"])
            except Exception: pass
import hashlib
seen, dedup = set(), []
for c in live:
    h = hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()
    if h not in seen: seen.add(h); dedup.append(c)
samp = dedup[:200]
mm = joblib.load(OUT)
t1 = time.time(); r = mm.predict_raw(samp[:100]); lat = (time.time() - t1) / 100 * 1000
raw = np.concatenate([r, mm.predict_raw(samp[100:])])
print(f"[{TAG}] latency {lat:.1f} ms/chunk | LIVE raw std={raw.std():.4f} range[{raw.min():.3f},{raw.max():.3f}] "
      f"uniq={len(np.unique(np.round(raw, 8)))}/{len(samp)}", flush=True)
print(f"[{TAG}] DONE", flush=True)
