#!/usr/bin/env python3
"""Merge parallel Gemma reasoning_eval shard JSON files into canonical outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "Stage2" / "reasoning_eval"

GROUPS = ("direct", "inference_augmented")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _base_name(task: str, model: str, group: str) -> str:
    subj = model.replace("/", "_")
    judge = "gemma-9b-it"
    return f"{task}_{judge}_{subj}_{group}.json"


def merge_one(base_path: Path, num_shards: int) -> int:
    merged_cases: dict = {}
    sources: list[str] = []

    if base_path.is_file():
        merged_cases.update(_load(base_path).get("cases", {}))
        sources.append(base_path.name)

    for i in range(num_shards):
        shard_path = base_path.parent / f"{base_path.stem}.shard{i}{base_path.suffix}"
        if not shard_path.is_file():
            continue
        shard_cases = _load(shard_path).get("cases", {})
        merged_cases.update(shard_cases)
        sources.append(shard_path.name)

    if not merged_cases:
        return 0

    doc = _load(base_path) if base_path.is_file() else {"meta": {}, "cases": {}}
    doc["cases"] = merged_cases
    meta = doc.setdefault("meta", {})
    ok = sum(1 for c in merged_cases.values() if c.get("status") == "ok")
    meta["merged_from"] = sources
    meta["completed_ok"] = ok
    _save(base_path, doc)
    print(f"  {base_path.name}: {ok} ok ({len(sources)} sources)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="diagnosis")
    parser.add_argument("--model", default="qwen3-14b-thinking")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    for group in GROUPS:
        base = args.out_dir / _base_name(args.task, args.model, group)
        merge_one(base, args.num_shards)


if __name__ == "__main__":
    main()
