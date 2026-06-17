#!/usr/bin/env python3
"""Gemma-9B reasoning_eval stratified by diagnosis accuracy (direct / aug)."""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RE = PROJECT_ROOT / "data" / "Stage1" / "reasoning_eval"
ACC_ROOT = PROJECT_ROOT / "data" / "Stage1" / "acc_results"
MET = ["efficiency", "factulity", "recall"]
LAB = ["Eff", "Fact", "Rec"]
SUBJ = ["o3-mini", "deepseek-r1", "qwen3-8b"]
JUDGE = "gemma-9b-it"


def load_reasoning(subj: str, group: str) -> dict[str, dict]:
    p = RE / f"diagnosis_{JUDGE}_{subj}_{group}.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {cid: row for cid, row in doc["cases"].items() if row.get("status") == "ok"}


def load_accuracy(subj: str) -> dict[str, bool]:
    d = ACC_ROOT / subj
    if not d.is_dir():
        return {}
    out: dict[str, bool] = {}
    for f in d.glob("*.json"):
        doc = json.loads(f.read_text(encoding="utf-8"))
        if "accuracy" in doc:
            out[f.stem] = bool(doc["accuracy"])
    return out


def mean_metric(rows: list[dict], key: str) -> float:
    return st.mean(r[key] for r in rows) if rows else float("nan")


def print_group(group: str) -> None:
    print("=" * 78)
    print(f"Gemma-9B · {group} · stratified by diagnosis accuracy (GPT judge)")
    print("=" * 78)

    missing_acc: list[str] = []
    for subj in SUBJ:
        acc = load_accuracy(subj)
        if len(acc) < 100:
            missing_acc.append(f"{subj}: {len(acc)}/100")

    if missing_acc:
        print("\nMissing accuracy labels:")
        for m in missing_acc:
            print(f"  {m}")
        print("\nRun first:")
        print("  python scripts/stage1/run_stage1_diagnosis_accuracy.py")
        return

    print()
    header = f"{'Model':14} {'Acc%':>6}  {'Layer':10} {'n':>4}  " + "  ".join(f"{l:>6}" for l in LAB)
    print(header)
    print("-" * len(header))

    for subj in SUBJ:
        re_rows = load_reasoning(subj, group)
        acc = load_accuracy(subj)
        common = sorted(set(re_rows) & set(acc))
        correct = [re_rows[c] for c in common if acc[c]]
        wrong = [re_rows[c] for c in common if not acc[c]]
        acc_rate = len(correct) / len(common) if common else float("nan")

        for layer_name, bucket in [("correct", correct), ("wrong", wrong), ("all", [re_rows[c] for c in common])]:
            if layer_name == "all":
                print(f"{subj:14} {acc_rate*100:5.1f}%  {'ALL':10} {len(bucket):4d}  ", end="")
            elif layer_name == "correct":
                print(f"{'':14} {'':>6}  {'correct':10} {len(bucket):4d}  ", end="")
            else:
                print(f"{'':14} {'':>6}  {'wrong':10} {len(bucket):4d}  ", end="")
            vals = "  ".join(f"{mean_metric(bucket, k)*100:5.1f}%" for k in MET)
            print(vals)

        if correct and wrong:
            print(f"{'':14} {'':>6}  {'d(c-w)':10} {'':4}  ", end="")
            deltas = "  ".join(
                f"{(mean_metric(correct, k) - mean_metric(wrong, k))*100:+5.1f}pp" for k in MET
            )
            print(deltas)
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        default="inference_augmented",
        choices=["direct", "inference_augmented", "both"],
    )
    args = parser.parse_args()
    groups = ["direct", "inference_augmented"] if args.group == "both" else [args.group]
    for i, g in enumerate(groups):
        if i:
            print()
        print_group(g)


if __name__ == "__main__":
    main()
