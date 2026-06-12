#!/usr/bin/env python3
"""Build Stage-1 demo subsets: 100 diagnosis + 100 treatment cases."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "MedRBench"

DEFAULT_DIAG_SRC = DATA_DIR / "diagnosis_957_cases_with_rare_disease_491.json"
DEFAULT_TREAT_SRC = DATA_DIR / "treatment_496_cases_with_rare_disease_165.json"
DEFAULT_TEST35 = DATA_DIR / "test_cases.json"
DEFAULT_DIAG_OUT = DATA_DIR / "demo_diagnosis_100.json"
DEFAULT_TREAT_OUT = DATA_DIR / "demo_treatment_100.json"
DEFAULT_MANIFEST_OUT = DATA_DIR / "demo_stage1_manifest.json"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _sample_ids(
    pool: dict,
    n: int,
    *,
    seed: int,
    must_include: list[str] | None = None,
    rare_key: str = "checked_rare_disease",
) -> list[str]:
    must_include = must_include or []
    missing = [cid for cid in must_include if cid not in pool]
    if missing:
        raise ValueError(f"Required case IDs not found in pool: {missing[:5]}")

    selected = list(dict.fromkeys(must_include))
    if len(selected) > n:
        raise ValueError(f"must_include has {len(selected)} IDs but n={n}")

    remaining_ids = [cid for cid in pool if cid not in selected]
    rare_ids = [cid for cid in remaining_ids if pool[cid].get(rare_key)]
    common_ids = [cid for cid in remaining_ids if not pool[cid].get(rare_key)]

    rng = random.Random(seed)
    target_rare = round((n - len(selected)) * len(rare_ids) / max(len(remaining_ids), 1))
    target_rare = min(target_rare, len(rare_ids), n - len(selected))

    pick_rare = rng.sample(rare_ids, target_rare) if target_rare else []
    selected.extend(pick_rare)

    still_need = n - len(selected)
    if still_need > 0:
        rest_pool = [cid for cid in remaining_ids if cid not in selected]
        selected.extend(rng.sample(rest_pool, still_need))

    return selected[:n]


def build_manifest(
    diag_ids: list[str],
    treat_ids: list[str],
    *,
    seed: int,
    diag_src: Path,
    treat_src: Path,
) -> dict:
    return {
        "stage": "stage1",
        "seed": seed,
        "counts": {"diagnosis": len(diag_ids), "treatment": len(treat_ids)},
        "diagnosis": {
            "source": str(diag_src.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "output": "data/MedRBench/demo_diagnosis_100.json",
            "case_ids": diag_ids,
        },
        "treatment": {
            "source": str(treat_src.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "output": "data/MedRBench/demo_treatment_100.json",
            "case_ids": treat_ids,
        },
        "subject_models": {
            "strong": [
                {"name": "o3-mini", "backend": "api", "note": "paper uses o3-mini; swap to o3 if API available"},
                {"name": "deepseek-r1", "backend": "api"},
            ],
            "weak": [
                {"name": "qwen3-8b", "backend": "local_vllm", "hf_model": "Qwen/Qwen3-8B"},
                {"name": "qwen3-14b", "backend": "local_vllm", "hf_model": "Qwen/Qwen3-14B"},
            ],
        },
        "gemma_scope_eval": {
            "base_models": ["google/gemma-2-2b", "google/gemma-2-9b"],
            "groups": {
                "direct": "Score reasoning text only (standard MedRBench metrics).",
                "sae_augmented": "Score reasoning + SAE feature summary from Gemma 2 forward pass.",
            },
            "sae_defaults": {
                "2b": {
                    "release": "gemma-scope-2b-pt-res-canonical",
                    "sae_id": "layer_12/width_16k/canonical",
                },
                "9b": {
                    "release": "gemma-scope-9b-pt-res-canonical",
                    "sae_id": "layer_12/width_16k/canonical",
                },
            },
        },
        "analysis_note": "Run all inference + scoring first; stratify strong/weak after full stats.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag-src", type=Path, default=DEFAULT_DIAG_SRC)
    parser.add_argument("--treat-src", type=Path, default=DEFAULT_TREAT_SRC)
    parser.add_argument("--test35", type=Path, default=DEFAULT_TEST35)
    parser.add_argument("--diag-out", type=Path, default=DEFAULT_DIAG_OUT)
    parser.add_argument("--treat-out", type=Path, default=DEFAULT_TREAT_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--diag-n", type=int, default=100)
    parser.add_argument("--treat-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-test35", action="store_true", default=True)
    parser.add_argument("--no-include-test35", dest="include_test35", action="store_false")
    args = parser.parse_args()

    diag_pool = _load(args.diag_src)
    treat_pool = _load(args.treat_src)
    test35 = _load(args.test35)

    must_include = list(test35.keys()) if args.include_test35 else []
    if args.diag_n < len(must_include):
        raise SystemExit(f"--diag-n ({args.diag_n}) must be >= test35 count ({len(must_include)})")

    diag_ids = _sample_ids(diag_pool, args.diag_n, seed=args.seed, must_include=must_include)
    treat_ids = _sample_ids(treat_pool, args.treat_n, seed=args.seed)

    diag_subset = {cid: diag_pool[cid] for cid in diag_ids}
    treat_subset = {cid: treat_pool[cid] for cid in treat_ids}

    _save(args.diag_out, diag_subset)
    _save(args.treat_out, treat_subset)
    _save(
        args.manifest_out,
        build_manifest(
            diag_ids,
            treat_ids,
            seed=args.seed,
            diag_src=args.diag_src,
            treat_src=args.treat_src,
        ),
    )

    diag_rare = sum(1 for cid in diag_ids if diag_pool[cid].get("checked_rare_disease"))
    treat_rare = sum(1 for cid in treat_ids if treat_pool[cid].get("checked_rare_disease"))
    print(f"Wrote {args.diag_out} ({len(diag_subset)} cases, rare={diag_rare})")
    print(f"Wrote {args.treat_out} ({len(treat_subset)} cases, rare={treat_rare})")
    print(f"Wrote {args.manifest_out}")
    if must_include:
        overlap = len(set(diag_ids) & set(test35))
        print(f"Included {overlap}/{len(test35)} test_cases.json IDs in diagnosis demo")


if __name__ == "__main__":
    main()
