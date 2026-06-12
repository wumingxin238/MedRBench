#!/usr/bin/env python3
"""Slice oracle / inference JSON to demo case IDs listed in stage1 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "MedRBench" / "demo_stage1_manifest.json"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def slice_outputs(src: dict, case_ids: list[str], models: list[str] | None = None) -> dict:
    out: dict = {}
    for cid in case_ids:
        if cid not in src:
            continue
        case_models = src[cid]
        if models is None:
            out[cid] = case_models
        else:
            out[cid] = {m: case_models[m] for m in models if m in case_models}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", choices=["diagnosis", "treatment"], required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model keys to keep, e.g. o3-mini deepseek-r1",
    )
    args = parser.parse_args()

    manifest = _load(args.manifest)
    case_ids = manifest[args.task]["case_ids"]
    src = _load(args.src)
    sliced = slice_outputs(src, case_ids, args.models)
    _save(args.out, sliced)
    print(f"Wrote {args.out}: {len(sliced)} cases")


if __name__ == "__main__":
    main()
