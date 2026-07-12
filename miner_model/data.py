"""Load harvested benchmark files into (hands, label, source_date) samples.

Each benchmark_<date>.json is a list of chunk-entries; every entry carries
`chunks` (list of batches, each batch = one player's hands) aligned with
`groundTruthLabels` ("human"/"bot"). The scoring unit is one batch.
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional, Tuple

Sample = Tuple[list, int, str]  # (hands, label 1=bot/0=human, source_date)


def _label_to_int(label) -> Optional[int]:
    value = label.get("is_bot") if isinstance(label, dict) else label
    text = str(value).strip().lower()
    if text in ("1", "true", "bot", "ai"):
        return 1
    if text in ("0", "false", "human"):
        return 0
    return None


def load_batches(data_dir: str, dates: Optional[set] = None) -> List[Sample]:
    samples: List[Sample] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "benchmark_*.json"))):
        date = os.path.basename(path)[len("benchmark_"):-len(".json")]
        if dates is not None and date not in dates:
            continue
        try:
            entries = json.load(open(path))
        except Exception:
            continue
        for entry in entries:
            batches = entry.get("chunks") or []
            labels = entry.get("groundTruthLabels") or entry.get("groundTruth") or []
            for hands, label in zip(batches, labels):
                y = _label_to_int(label)
                if y is None or not isinstance(hands, list) or not hands:
                    continue
                samples.append((hands, y, date))
    return samples


def available_dates(data_dir: str) -> List[str]:
    dates = []
    for path in sorted(glob.glob(os.path.join(data_dir, "benchmark_*.json"))):
        dates.append(os.path.basename(path)[len("benchmark_"):-len(".json")])
    return sorted(dates)
