"""Behavioral feature engineering for one chunk (a batch of a player's hands).

Signal is betting-behavior only (no cards / results / timing). On top of the
champion's core (action shares, marginal entropy, run lengths, pot dynamics,
distributional aggregation, action n-grams) this adds three differentiators:

  * conditional/transition entropy of the action stream  (sequential predictability)
  * bet-size quantization fingerprint                    (bots snap to few sizes)
  * cross-hand self-similarity                           (bots repeat themselves)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

_ACTIONS = ["fold", "call", "check", "bet", "raise"]
_CODE = {"fold": "F", "call": "C", "check": "K", "bet": "B", "raise": "R"}
_LETTERS = "FCKBR"
_BIGRAMS = [a + b for a in _LETTERS for b in _LETTERS]  # 25 transition types
_ROUND_SIZES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]
_SIZE_EDGES = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def _entropy(counts: List[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total + 1e-12, 2) for c in counts if c > 0)


def _norm_action(action: dict):
    action_type = str(action.get("action_type", "")).lower()
    return action_type if action_type in _ACTIONS else None


def _transition_entropy(types: List[str]) -> float:
    """H(next action | previous action) — sequential predictability (low = bot-like)."""
    if len(types) < 2:
        return 0.0
    trans = defaultdict(Counter)
    for prev, nxt in zip(types[:-1], types[1:]):
        trans[prev][nxt] += 1
    total = len(types) - 1
    h = 0.0
    for _prev, nexts in trans.items():
        p_prev = sum(nexts.values()) / total
        h += p_prev * _entropy(list(nexts.values()))
    return h


def _max_run(types: List[str]) -> int:
    if not types:
        return 0
    best = cur = 1
    for i in range(1, len(types)):
        cur = cur + 1 if types[i] == types[i - 1] else 1
        best = max(best, cur)
    return best


def _hand_features(hand: dict) -> Tuple[Dict[str, float], List[str], List[float]]:
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)]
    types = [t for t in (_norm_action(a) for a in actions) if t]
    n = max(1, len(types))
    counts = Counter(types)
    feat: Dict[str, float] = {}
    for a in _ACTIONS:
        feat[f"{a}_share"] = counts.get(a, 0) / n
    feat["aggr_share"] = (counts.get("bet", 0) + counts.get("raise", 0)) / n
    feat["passive_share"] = (counts.get("call", 0) + counts.get("check", 0)) / n
    feat["n_actions"] = float(len(types))
    feat["n_streets"] = float(len(hand.get("streets") or []))
    feat["n_players"] = float(len(hand.get("players") or []))

    amounts = [float(a.get("normalized_amount_bb") or 0.0) for a in actions]
    nonzero = [x for x in amounts if x > 0]
    feat["amt_mean"] = float(np.mean(nonzero)) if nonzero else 0.0
    feat["amt_std"] = float(np.std(nonzero)) if len(nonzero) > 1 else 0.0
    feat["amt_max"] = float(max(nonzero)) if nonzero else 0.0
    feat["nonzero_share"] = len(nonzero) / n

    pots_after = [float(a.get("pot_after") or 0.0) for a in actions]
    pots_before = [float(a.get("pot_before") or 0.0) for a in actions]
    if pots_after and pots_before:
        feat["pot_growth"] = max(0.0, pots_after[-1] - pots_before[0])
        mono = sum(1 for i in range(1, len(pots_after)) if pots_after[i] >= pots_after[i - 1])
        feat["pot_monotonic"] = mono / max(1, len(pots_after) - 1)
    else:
        feat["pot_growth"] = 0.0
        feat["pot_monotonic"] = 0.0

    feat["action_entropy"] = _entropy(list(counts.values()))
    feat["transition_entropy"] = _transition_entropy(types)
    feat["max_run_share"] = _max_run(types) / n
    return feat, types, nonzero


def _quantization_features(nonzero: List[float]) -> Dict[str, float]:
    """Bots snap to a few discrete/round bet sizes; humans spread out."""
    if not nonzero:
        return {"q_unique_ratio": 0.0, "q_mode_conc": 0.0, "q_round_share": 0.0, "q_bet_std": 0.0, "q_bet_cv": 0.0}
    arr = np.asarray(nonzero, dtype=float)
    rounded = np.round(arr, 2)
    counts = Counter(rounded.tolist())
    mean = float(arr.mean())
    std = float(arr.std())
    round_share = float(np.mean([any(abs(x - r) <= 0.06 for r in _ROUND_SIZES) for x in arr]))
    return {
        "q_unique_ratio": len(counts) / len(arr),
        "q_mode_conc": max(counts.values()) / len(arr),
        "q_round_share": round_share,
        "q_bet_std": std,
        "q_bet_cv": (std / mean) if mean > 0 else 0.0,
    }


def _self_similarity(vectors: List[List[float]]) -> Dict[str, float]:
    """Mean pairwise distance between per-hand action-share vectors (low = self-similar = bot-like)."""
    if len(vectors) < 2:
        return {"selfsim_mean_dist": 0.0, "selfsim_share_std": 0.0}
    matrix = np.asarray(vectors, dtype=float)
    sample = matrix
    if len(matrix) > 60:  # cap pairwise cost
        idx = np.linspace(0, len(matrix) - 1, 60).astype(int)
        sample = matrix[idx]
    dists: List[float] = []
    for i in range(len(sample)):
        if i + 1 < len(sample):
            dists.extend(np.sqrt(((sample[i] - sample[i + 1:]) ** 2).sum(axis=1)).tolist())
    return {
        "selfsim_mean_dist": float(np.mean(dists)) if dists else 0.0,
        "selfsim_share_std": float(matrix.std(axis=0).mean()),
    }


def _bigram_features(all_types: List[List[str]]) -> Dict[str, float]:
    counts: Counter = Counter()
    total = 0
    for types in all_types:
        codes = [_CODE[t] for t in types]
        for a, b in zip(codes[:-1], codes[1:]):
            counts[a + b] += 1
            total += 1
    return {f"bg_{bg}": (counts.get(bg, 0) / total if total else 0.0) for bg in _BIGRAMS}


def _street_regularity(hands: List[dict]) -> Dict[str, float]:
    """Per-street aggression + how rigidly the player repeats it across streets.

    Bots often play every street the same mechanical way (low variance across
    streets); humans adapt preflop vs postflop. Low rigidity = bot-like.
    """
    tallies = {"preflop": [0, 0], "flop": [0, 0], "turn": [0, 0], "river": [0, 0]}  # [aggressive, total]
    for hand in hands:
        for action in (hand.get("actions") or []):
            if not isinstance(action, dict):
                continue
            atype = str(action.get("action_type", "")).lower()
            street = str(action.get("street", "")).lower()
            if atype not in _ACTIONS or street not in tallies:
                continue
            tallies[street][1] += 1
            if atype in ("bet", "raise"):
                tallies[street][0] += 1
    shares = {s: (agg / tot if tot else 0.0) for s, (agg, tot) in tallies.items()}
    postflop = [shares["flop"], shares["turn"], shares["river"]]
    return {
        "street_aggr_preflop": shares["preflop"],
        "street_aggr_flop": shares["flop"],
        "street_aggr_turn": shares["turn"],
        "street_aggr_river": shares["river"],
        "street_aggr_rigidity": float(np.std(list(shares.values()))),        # low = same every street
        "preflop_postflop_gap": abs(shares["preflop"] - float(np.mean(postflop))),
    }


def _sequence_repetition(all_types: List[List[str]]) -> Dict[str, float]:
    """Do the player's hands repeat identical action sequences? Bots do; humans vary."""
    seqs = ["".join(_CODE[t] for t in types) for types in all_types if types]
    if not seqs:
        return {"seq_repeat_rate": 0.0, "seq_unique_ratio": 0.0}
    counts = Counter(seqs)
    return {
        "seq_repeat_rate": max(counts.values()) / len(seqs),   # share on the single most common sequence
        "seq_unique_ratio": len(counts) / len(seqs),           # low = repetitive = bot-like
    }


def _extreme_hand_rates(per_hand: List[Dict[str, float]]) -> Dict[str, float]:
    """Fraction of a player's hands that are 'extreme' (mechanically bot-like).

    All three current top-3 miners independently use rate-of-extreme-hand features —
    they capture *how often* a player behaves like a bot, not just the average.
    """
    n = len(per_hand)
    if n == 0:
        return {"low_entropy_hand_rate": 0.0, "high_aggression_hand_rate": 0.0,
                "long_hand_rate": 0.0, "low_transition_hand_rate": 0.0}
    return {
        "low_entropy_hand_rate": sum(f["action_entropy"] < 0.5 for f in per_hand) / n,
        "high_aggression_hand_rate": sum(f["aggr_share"] > 0.5 for f in per_hand) / n,
        "long_hand_rate": sum(f["n_actions"] >= 6 for f in per_hand) / n,
        "low_transition_hand_rate": sum(f["transition_entropy"] < 0.5 for f in per_hand) / n,
    }


def _bet_pot_features(hands: List[dict]) -> Dict[str, float]:
    """Bet size relative to the pot. Bots bet consistent pot fractions (tight clustering)."""
    ratios = []
    for hand in hands:
        for action in (hand.get("actions") or []):
            if not isinstance(action, dict):
                continue
            if str(action.get("action_type", "")).lower() not in ("bet", "raise"):
                continue
            amount = float(action.get("normalized_amount_bb") or 0.0)
            pot = float(action.get("pot_before") or 0.0)
            if amount > 0 and pot > 0:
                ratios.append(min(amount / pot, 5.0))
    if not ratios:
        return {"bet_pot_mean": 0.0, "bet_pot_std": 0.0, "bet_pot_cv": 0.0, "bet_pot_mode_frac": 0.0}
    arr = np.asarray(ratios, dtype=float)
    mean, std = float(arr.mean()), float(arr.std())
    quarter = np.round(arr * 4) / 4.0  # cluster to nearest 0.25 pot-fraction
    return {
        "bet_pot_mean": mean,
        "bet_pot_std": std,
        "bet_pot_cv": std / mean if mean > 0 else 0.0,
        "bet_pot_mode_frac": max(Counter(quarter.tolist()).values()) / len(arr),
    }


def _distribution_entropy_features(all_types: List[List[str]], all_nonzero: List[float]) -> Dict[str, float]:
    """Single summary stats of how predictable the action-transitions and bet-sizes are."""
    bigrams: Counter = Counter()
    for types in all_types:
        codes = [_CODE[t] for t in types]
        for a, b in zip(codes[:-1], codes[1:]):
            bigrams[a + b] += 1
    bigram_entropy = _entropy(list(bigrams.values())) if bigrams else 0.0
    if all_nonzero:
        counts = Counter(np.digitize(all_nonzero, _SIZE_EDGES).tolist())
        size_entropy = _entropy(list(counts.values()))
        size_modal = max(counts.values()) / len(all_nonzero)
    else:
        size_entropy, size_modal = 0.0, 0.0
    return {"action_bigram_entropy": bigram_entropy,
            "size_bucket_entropy": size_entropy, "size_modal_frac": size_modal}


def batch_features(hands: List[dict]) -> Dict[str, float]:
    """Featurize one chunk (a batch of hands) into a flat dict of numeric features."""
    hands = [h for h in hands if isinstance(h, dict)]
    if not hands:
        return {"hand_count": 0.0}

    per_hand: List[Dict[str, float]] = []
    all_types: List[List[str]] = []
    all_nonzero: List[float] = []
    share_vectors: List[List[float]] = []
    for hand in hands:
        feat, types, nonzero = _hand_features(hand)
        per_hand.append(feat)
        all_types.append(types)
        all_nonzero.extend(nonzero)
        share_vectors.append([feat[f"{a}_share"] for a in _ACTIONS])

    out: Dict[str, float] = {"hand_count": float(len(hands))}
    for key in per_hand[0].keys():  # distributional aggregation across hands
        vals = np.asarray([p[key] for p in per_hand], dtype=float)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
        out[f"{key}_q10"] = float(np.quantile(vals, 0.10))
        out[f"{key}_q50"] = float(np.quantile(vals, 0.50))
        out[f"{key}_q90"] = float(np.quantile(vals, 0.90))
    out.update(_quantization_features(all_nonzero))
    out.update(_self_similarity(share_vectors))
    out.update(_bigram_features(all_types))
    out.update(_street_regularity(hands))       # separation-widening: street rigidity
    out.update(_sequence_repetition(all_types))  # separation-widening: repeated action sequences
    out.update(_extreme_hand_rates(per_hand))    # convergent (top-3): rate of bot-like hands
    out.update(_bet_pot_features(hands))         # convergent (top-3): bet/pot-ratio clustering
    out.update(_distribution_entropy_features(all_types, all_nonzero))  # bigram + size-bucket entropy
    return out
