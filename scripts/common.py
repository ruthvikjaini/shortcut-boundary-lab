"""Shared helpers for the run scripts: paths, JSON output, aggregation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CKPT = RESULTS / "checkpoints"


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    print(f"wrote {path}")


def mean3(values):
    """Mean of 3 seeds, rounded to 4dp, with the per-seed values."""
    vals = [round(float(v), 4) for v in values]
    return {"mean": round(statistics.fmean(vals), 4), "seeds": vals}
