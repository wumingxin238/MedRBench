#!/usr/bin/env python3
"""Merge parallel inference shard JSON files into one output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFERENCE_DIR = PROJECT_ROOT / "data" / "Stage2" / "inference"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="e.g. qwen3-14b-thinking")
    parser.add_argument("--task", choices=["diagnosis", "treatment"], required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--inference-dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or args.inference_dir / f"{args.model}_{args.task}.json"
    merged: dict = {}
    missing: list[int] = []

    for i in range(args.num_shards):
        shard_path = args.inference_dir / f"{args.model}_{args.task}.shard{i}.json"
        if not shard_path.is_file():
            missing.append(i)
            continue
        data = json.loads(shard_path.read_text(encoding="utf-8"))
        overlap = set(merged) & set(data)
        if overlap:
            print(f"Warning: shard {i} overlaps {len(overlap)} case ids with prior shards")
        merged.update(data)
        print(f"  shard {i}: +{len(data)} cases ({shard_path.name})")

    if missing:
        print(f"Missing shards (not merged yet): {missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} ({len(merged)} cases total)")


if __name__ == "__main__":
    main()
