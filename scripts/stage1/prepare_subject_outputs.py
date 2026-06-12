#!/usr/bin/env python3
"""
Merge Stage-1 subject model outputs for Gemma Scope evaluation.

Strong models (o3-mini, deepseek-r1): sliced from full oracle JSON.
Weak models (qwen3-8b, qwen3-14b): from data/Stage1/inference/*.json after local runs.

Example:
  python scripts/stage1/prepare_subject_outputs.py --task diagnosis
  python scripts/stage1/prepare_subject_outputs.py --task treatment \\
      --strong-src path/to/oracle_treatment.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "MedRBench" / "demo_stage1_manifest.json"
STAGE1_DIR = PROJECT_ROOT / "data" / "Stage1"
INFERENCE_DIR = STAGE1_DIR / "inference"

DEFAULT_STRONG_SRC = {
    "diagnosis": PROJECT_ROOT / "oracle_diagnosis.json",
    "treatment": None,
}

DEFAULT_CASES = {
    "diagnosis": PROJECT_ROOT / "data" / "MedRBench" / "demo_diagnosis_100.json",
    "treatment": PROJECT_ROOT / "data" / "MedRBench" / "demo_treatment_100.json",
}

DEFAULT_OUT = {
    "diagnosis": STAGE1_DIR / "oracle_diagnosis_subjects.json",
    "treatment": STAGE1_DIR / "oracle_treatment_subjects.json",
}

STRONG_MODELS = ["o3-mini", "deepseek-r1"]
WEAK_MODELS = ["qwen3-8b", "qwen3-14b"]


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _slice_models(src: dict, case_ids: list[str], models: list[str]) -> dict:
    out: dict = {}
    for cid in case_ids:
        if cid not in src:
            continue
        entry = src[cid]
        picked = {m: entry[m] for m in models if m in entry}
        if picked:
            out[cid] = picked
    return out


def _merge_weak(base: dict, weak_path: Path, model_key: str) -> int:
    if not weak_path.is_file():
        return 0
    weak = _load(weak_path)
    added = 0
    for cid, models in weak.items():
        if model_key not in models:
            continue
        base.setdefault(cid, {})[model_key] = models[model_key]
        added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["diagnosis", "treatment"], required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--strong-src", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest = _load(args.manifest)
    case_ids = manifest[args.task]["case_ids"]
    cases_path = args.cases or DEFAULT_CASES[args.task]
    out_path = args.out or DEFAULT_OUT[args.task]
    strong_src = args.strong_src or DEFAULT_STRONG_SRC[args.task]

    merged: dict = {}

    if strong_src and strong_src.is_file():
        oracle = _load(strong_src)
        merged = _slice_models(oracle, case_ids, STRONG_MODELS)
        print(f"Strong models from {strong_src}: {len(merged)} cases")
    elif strong_src:
        print(f"Warning: strong oracle not found: {strong_src}")

    for model in WEAK_MODELS:
        weak_path = INFERENCE_DIR / f"{model}_{args.task}.json"
        n = _merge_weak(merged, weak_path, model)
        print(f"  {model}: +{n} cases from {weak_path.name}")

    _save(out_path, merged)

    # Coverage report
    cases = _load(cases_path)
    all_models = STRONG_MODELS + WEAK_MODELS
    print(f"\nWrote {out_path} ({len(merged)} cases)")
    for model in all_models:
        have = sum(1 for cid in case_ids if cid in merged and model in merged[cid])
        print(f"  {model}: {have}/{len(case_ids)}")


if __name__ == "__main__":
    main()
