#!/usr/bin/env python3
"""
Build Stage-2 diagnosis set: existing demo 100 + 300 hard cases from remaining 857.

Difficulty score (higher = harder), using available 957-run artifacts:
  - gemini2-ft accuracy wrong (+3)
  - deepseek-r1 / qwq low completeness recall (+0..2 from 1-recall)
  - cross-model recall spread deepseek vs qwq (+spread)
  - demo-100: Stage-1 wrong diagnosis count across o3/deepseek/qwen3-8b (+2 per wrong)
  - demo-100: Gemma-9B cross-model reasoning spread (+spread)

Outputs:
  data/MedRBench/hard_diagnosis_300.json
  data/MedRBench/diagnosis_400.json
  data/MedRBench/stage2_manifest.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "MedRBench"
STAGE1 = PROJECT_ROOT / "data" / "Stage1"
EVAL = PROJECT_ROOT / "src" / "Evaluation"

DEFAULT_DEMO_MANIFEST = DATA / "demo_stage1_manifest.json"
DEFAULT_POOL = DATA / "diagnosis_957_cases_with_rare_disease_491.json"
DEFAULT_HARD_OUT = DATA / "hard_diagnosis_300.json"
DEFAULT_MERGED_OUT = DATA / "diagnosis_400.json"
DEFAULT_MANIFEST_OUT = DATA / "stage2_manifest.json"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _acc_wrong(acc_dir: Path, cid: str) -> bool | None:
    p = acc_dir / f"{cid}.json"
    if not p.is_file():
        return None
    return not bool(_load(p).get("accuracy"))


def _reasoning_recall(reason_dir: Path, cid: str) -> float | None:
    p = reason_dir / f"{cid}.json"
    if not p.is_file():
        return None
    return float(_load(p).get("recall", 0) or 0)


def _demo_difficulty(demo_ids: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {cid: 0.0 for cid in demo_ids}
    # accuracy wrong count
    for model in ["o3-mini", "deepseek-r1", "qwen3-8b"]:
        acc_dir = STAGE1 / "acc_results" / model
        for cid in demo_ids:
            w = _acc_wrong(acc_dir, cid)
            if w:
                scores[cid] += 2.0
    # 9B reasoning spread across o3/deepseek/qwen
    by: dict[str, dict[str, float]] = {m: {} for m in ["o3-mini", "deepseek-r1", "qwen3-8b"]}
    for m in by:
        for group in ["direct", "inference_augmented"]:
            p = STAGE1 / "reasoning_eval" / f"diagnosis_gemma-9b-it_{m}_{group}.json"
            if not p.is_file():
                continue
            doc = _load(p)
            for cid, row in doc.get("cases", {}).items():
                if row.get("status") == "ok" and cid in scores:
                    by[m][cid] = row.get("recall", 0)
    for cid in demo_ids:
        vals = [by[m][cid] for m in by if cid in by[m]]
        if len(vals) >= 2:
            scores[cid] += max(vals) - min(vals)
    return scores


def _pool_difficulty(pool_ids: list[str]) -> dict[str, float]:
    gem_acc = EVAL / "acc_results_qwen_judge_paper_957" / "gemini2-ft"
    ds_re = EVAL / "reasoning_results_qwen_judge_paper_957" / "deepseek-r1"
    qwq_re = EVAL / "reasoning_results_qwen_judge_paper_957" / "qwq"

    scores: dict[str, float] = {}
    for cid in pool_ids:
        s = 0.0
        w = _acc_wrong(gem_acc, cid)
        if w is True:
            s += 3.0
        elif w is False:
            s += 0.5  # correct but still pool candidate

        ds_rec = _reasoning_recall(ds_re, cid)
        qwq_rec = _reasoning_recall(qwq_re, cid)
        for rec in (ds_rec, qwq_rec):
            if rec is not None:
                s += max(0.0, 1.0 - rec) * 1.5
        if ds_rec is not None and qwq_rec is not None:
            s += abs(ds_rec - qwq_rec) * 2.0

        scores[cid] = s
    return scores


def build_manifest(demo_ids: list[str], hard_ids: list[str], *, seed: int) -> dict:
    all_ids = demo_ids + hard_ids
    return {
        "stage": "stage2",
        "seed": seed,
        "counts": {"diagnosis": len(all_ids), "hard_only": len(hard_ids), "demo_reuse": len(demo_ids)},
        "diagnosis": {
            "source": "data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json",
            "demo_source": "data/MedRBench/demo_diagnosis_100.json",
            "hard_output": "data/MedRBench/hard_diagnosis_300.json",
            "merged_output": "data/MedRBench/diagnosis_400.json",
            "case_ids": all_ids,
            "demo_case_ids": demo_ids,
            "hard_case_ids": hard_ids,
        },
        "subject_models": {
            "weak": [
                {
                    "name": "qwen3-14b-thinking",
                    "backend": "local",
                    "hf_model": "Qwen/Qwen3-14B-AWQ",
                    "enable_thinking": True,
                    "note": "Same Qwen3-14B checkpoint; thinking mode via chat template",
                }
            ],
        },
        "eval": {
            "reasoning_judge": "gemma-9b-it",
            "reasoning_groups": ["direct", "inference_augmented"],
            "accuracy_judge": "gpt-5",
            "accuracy_env": "EVAL_MODEL=gpt-5 (or your API alias)",
        },
        "note": "Demo 100 subject/oracle/gemma9b results are reused; only qwen3-14b-thinking runs on 400.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-manifest", type=Path, default=DEFAULT_DEMO_MANIFEST)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--hard-n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-out", type=Path, default=DEFAULT_HARD_OUT)
    parser.add_argument("--merged-out", type=Path, default=DEFAULT_MERGED_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--scores-out", type=Path, default=DATA / "stage2_hard_scores.json")
    args = parser.parse_args()

    demo_manifest = _load(args.demo_manifest)
    pool = _load(args.pool)
    demo_ids = list(demo_manifest["diagnosis"]["case_ids"])
    demo_set = set(demo_ids)

    remaining = [cid for cid in pool if cid not in demo_set]
    if len(remaining) < args.hard_n:
        raise SystemExit(f"Only {len(remaining)} cases outside demo; need {args.hard_n}")

    demo_scores = _demo_difficulty(demo_ids)
    pool_scores = _pool_difficulty(remaining)

    ranked = sorted(remaining, key=lambda c: (-pool_scores[c], c))
    hard_ids = ranked[: args.hard_n]

    hard_subset = {cid: pool[cid] for cid in hard_ids}
    merged_ids = demo_ids + hard_ids
    merged_subset = {cid: pool[cid] for cid in merged_ids}

    _save(args.hard_out, hard_subset)
    _save(args.merged_out, merged_subset)
    _save(args.manifest_out, build_manifest(demo_ids, hard_ids, seed=args.seed))

    score_report = {
        "hard_selected": [
            {"id": cid, "pool_score": round(pool_scores[cid], 4)} for cid in hard_ids[:20]
        ],
        "hard_tail": [
            {"id": cid, "pool_score": round(pool_scores[cid], 4)} for cid in hard_ids[-5:]
        ],
        "demo_scores_sample": [
            {"id": cid, "score": round(demo_scores[cid], 4)} for cid in sorted(demo_scores, key=lambda x: -demo_scores[x])[:10]
        ],
    }
    _save(args.scores_out, score_report)

    hard_rare = sum(1 for cid in hard_ids if pool[cid].get("checked_rare_disease"))
    print(f"Demo reuse: {len(demo_ids)} cases (no re-inference for strong / qwen3-8b)")
    print(f"Hard 300: {args.hard_out}  rare={hard_rare}/{len(hard_ids)} ({100*hard_rare/len(hard_ids):.0f}%)")
    print(f"Merged 400: {args.merged_out}")
    print(f"Manifest: {args.manifest_out}")
    print(f"Top hard score: {pool_scores[hard_ids[0]]:.3f}  bottom selected: {pool_scores[hard_ids[-1]]:.3f}")


if __name__ == "__main__":
    main()
