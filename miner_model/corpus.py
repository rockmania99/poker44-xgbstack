"""Human-hands corpus loader: real labeled-human training chunks.

The canonical subnet repo ships ~32k REAL human poker hands
(hands_generator/human_hands/poker_hands_combined.json.gz) as public training
material. Benchmark-only models can *invert* on real humans (rank them above
bots); adding corpus-human chunks fixes that (proven on this subnet, 2026-07-04
research line). Hands are RAW here — every hand must pass through the
validator's own `prepare_hand_for_miner()` projection so training matches what
miners actually see.

Chunks are built per-player (hero) in original sequential order, mirroring the
live "one chunk = one player's session" unit. Each player's timeline is
partitioned into disjoint regions so the three outputs never share a hand:
  [ 0..70% -> small train chunks | 70..90% -> large train chunks | 90..100% -> holdout ]
"""
from __future__ import annotations

import gzip
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from poker44.validator.payload_view import prepare_hand_for_miner

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(
    _PROJECT_ROOT, "hands_generator", "human_hands", "poker_hands_combined.json.gz"
)


def _hero_uid(hand: dict) -> str:
    hero_seat = (hand.get("metadata") or {}).get("hero_seat")
    for p in hand.get("players") or []:
        if p.get("seat") == hero_seat:
            return str(p.get("player_uid") or "unknown")
    return "unknown"


def _cut(seq: List[dict], size_min: int, size_max: int, rng: random.Random) -> List[list]:
    out, i = [], 0
    while i + size_min <= len(seq):
        size = rng.randint(size_min, size_max)
        chunk = seq[i:i + size]
        if len(chunk) >= size_min:
            out.append(chunk)
        i += size
    return out


def load_corpus_chunks(
    *,
    small: Tuple[int, int] = (30, 40),
    large: Tuple[int, int] = (80, 105),
    max_small: int = 600,
    max_large: int = 150,
    max_holdout: int = 60,
    seed: int = 42,
    path: str = CORPUS_PATH,
) -> Tuple[List[list], List[list], List[list]]:
    """Return (small_train, large_train, holdout) sanitized human chunks."""
    with gzip.open(path, "rt") as fh:
        hands = json.load(fh)

    by_player: Dict[str, List[dict]] = defaultdict(list)
    for h in hands:
        by_player[_hero_uid(h)].append(h)

    rng = random.Random(seed)
    small_train: List[list] = []
    large_train: List[list] = []
    holdout: List[list] = []
    for uid in sorted(by_player):
        seq = by_player[uid]
        a, b = int(len(seq) * 0.70), int(len(seq) * 0.90)
        small_train += _cut(seq[:a], *small, rng)
        large_train += _cut(seq[a:b], *large, rng)
        holdout += _cut(seq[b:], *small, rng)

    def _sanitize(chunks: List[list]) -> List[list]:
        return [[prepare_hand_for_miner(h) for h in c] for c in chunks]

    rng.shuffle(small_train)
    rng.shuffle(large_train)
    rng.shuffle(holdout)
    return (
        _sanitize(small_train[:max_small]),
        _sanitize(large_train[:max_large]),
        _sanitize(holdout[:max_holdout]),
    )


def bootstrap_to_size(
    chunk: list, *, size_min: int = 80, size_max: int = 105, rng: random.Random
) -> list:
    """Resample a small chunk (with replacement) up to live size (live chunks are
    pinned to 100 hands). Used to give BOTH classes large-size training examples
    so chunk size can't become a label shortcut. Introduces artificial duplicate
    hands — applied equally to both classes."""
    size = rng.randint(size_min, size_max)
    return [chunk[rng.randrange(len(chunk))] for _ in range(size)]
