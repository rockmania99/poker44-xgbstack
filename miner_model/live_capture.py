"""Fail-safe capture of incoming validator chunks (UNLABELED) for offline
feature-distribution analysis — never affects serving, never sees labels.

Purpose: per-feature stability screening (keep features that carry benchmark
signal AND do not shift on the live feed). This is the field-proven answer to
the benchmark->live distribution gap. Captured data is used for unsupervised
distribution diagnostics / feature selection only — never as training labels.

Env: POKER44_CAPTURE=1 to enable. Files: data/live_chunks/<tag>_<ts>.json.gz,
hard cap on file count; every failure is swallowed.
"""
from __future__ import annotations

import gzip
import json
import os
import time


def capture_chunks(chunks, tag: str, base_dir: str) -> None:
    try:
        if os.getenv("POKER44_CAPTURE", "0") != "1" or not chunks:
            return
        d = os.path.join(base_dir, "data", "live_chunks")
        os.makedirs(d, exist_ok=True)
        existing = [f for f in os.listdir(d) if f.startswith(tag)]
        if len(existing) >= int(os.getenv("POKER44_CAPTURE_MAX_FILES", "200")):
            return
        path = os.path.join(d, f"{tag}_{int(time.time())}.json.gz")
        with gzip.open(path, "wt") as fh:
            json.dump({"t": time.time(), "n": len(chunks),
                       "sizes": [len(c or []) for c in chunks],
                       "chunks": chunks}, fh)
    except Exception:
        pass  # capture must never break serving
