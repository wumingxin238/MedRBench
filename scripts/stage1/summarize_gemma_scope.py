#!/usr/bin/env python3
"""Summarize Stage-1 Gemma Scope eval JSON (direct vs sae_augmented)."""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data" / "Stage1" / "gemma_scope"


def _parse_stats(cases: dict, group: str) -> tuple[list[int], int, int]:
    scores: list[int] = []
    failed = 0
    for c in cases.values():
        p = c.get("scores", {}).get(group, {}).get("parsed")
        if p and p.get("score") is not None:
            scores.append(int(p["score"]))
        else:
            failed += 1
    return scores, len(scores), failed


def summarize_file(path: Path) -> None:
    d = json.load(path.open(encoding="utf-8"))
    cases = d.get("cases", {})
    meta = d.get("meta", {})
    print(f"\n{'=' * 60}")
    print(path.name)
    print(f"  gemma: {meta.get('gemma_model', '?')}  subject: {meta.get('subject_model', '?')}")
    print(f"  cases: {len(cases)}")
    means: dict[str, float] = {}
    for g in ("direct", "sae_augmented"):
        scores, ok, fail = _parse_stats(cases, g)
        if not scores:
            print(f"  {g}: parsed 0/100  failed={fail}")
            continue
        means[g] = st.mean(scores)
        print(
            f"  {g}: parsed {ok}/100  failed={fail}  "
            f"mean={st.mean(scores):.2f}  median={st.median(scores):.1f}  "
            f"dist={dict(sorted(Counter(scores).items()))}"
        )
    if len(means) == 2:
        print(f"  delta (sae - direct): {means['sae_augmented'] - means['direct']:+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    files = sorted(args.dir.glob("diagnosis_*.json"))
    if not files:
        print(f"No files in {args.dir}")
        return

    subj_path = args.dir.parent / "oracle_diagnosis_subjects.json"
    if subj_path.is_file():
        subj = json.load(subj_path.open(encoding="utf-8"))
        print("Subject coverage:")
        for m in ("deepseek-r1", "o3-mini", "qwen3-8b", "qwen3-14b"):
            n = sum(1 for v in subj.values() if m in v)
            print(f"  {m}: {n}/100")

    inf = args.dir.parent / "inference" / "qwen3-8b_diagnosis.json"
    if inf.is_file():
        n = len(json.load(inf.open(encoding="utf-8")))
        print(f"Qwen3-8B inference: {n}/100")

    for f in files:
        summarize_file(f)

    by_subject: dict[str, dict[str, dict]] = {}
    for f in files:
        d = json.load(f.open(encoding="utf-8"))
        subject = d.get("meta", {}).get("subject_model") or f.stem.split("_", 2)[-1]
        size = d.get("meta", {}).get("gemma_size") or ("9b" if "_9b_" in f.name else "2b")
        by_subject.setdefault(subject, {})[size] = d

    if any(len(v) > 1 for v in by_subject.values()):
        print("\n" + "=" * 60)
        print("2B vs 9B (direct mean, parsed only)")
        for subject, sizes in sorted(by_subject.items()):
            if "2b" not in sizes or "9b" not in sizes:
                continue
            row = [subject]
            for size in ("2b", "9b"):
                scores, ok, _ = _parse_stats(sizes[size].get("cases", {}), "direct")
                row.append(f"{size}={st.mean(scores):.2f}({ok})" if scores else f"{size}=n/a")
            print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
