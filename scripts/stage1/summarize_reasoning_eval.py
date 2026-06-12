#!/usr/bin/env python3
"""Summarize Stage-1 MedRBench-style reasoning eval JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data" / "Stage1" / "reasoning_eval"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.dir.glob("*.json")):
        doc = _load(path)
        meta = doc.get("meta", {})
        rows.append(
            {
                "file": path.name,
                "task": meta.get("task"),
                "subject": meta.get("subject_model"),
                "group": meta.get("group"),
                "judge": meta.get("judge_model"),
                "n_ok": meta.get("completed_ok", 0),
                "efficiency": meta.get("mean_efficiency"),
                "factulity": meta.get("mean_factulity"),
                "recall": meta.get("mean_recall"),
            }
        )

    if not rows:
        print(f"No JSON in {args.dir}")
        return

    print(f"{'file':<45} {'eff':>7} {'fact':>7} {'recall':>7}  n")
    print("-" * 75)
    def _fmt(v):
        return f"{v:>7.3f}" if v is not None else f"{'—':>7}"

    for r in rows:
        print(
            f"{r['file']:<45} {_fmt(r['efficiency'])} {_fmt(r['factulity'])} "
            f"{_fmt(r['recall'])} {r['n_ok']:>3}"
        )


if __name__ == "__main__":
    main()
