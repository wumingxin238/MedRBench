#!/usr/bin/env python3
"""Verify report sections not covered by verify_report_numbers.py."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RE = ROOT / "data" / "Stage1" / "reasoning_eval"
SCOPE = ROOT / "data" / "Stage1" / "gemma_scope"
SUBJ = ["o3-mini", "deepseek-r1", "qwen3-8b"]


def scope_mean(path: Path, group: str) -> tuple[int, float | None]:
    d = json.loads(path.read_text(encoding="utf-8"))
    scores = []
    for c in d["cases"].values():
        sc = c.get("scores", {}).get(group, {}).get("parsed", {}).get("score")
        if sc is not None:
            scores.append(sc)
    return len(scores), round(st.mean(scores), 2) if scores else None


def main() -> None:
    issues: list[str] = []

    # §2 rare
    m = json.loads((ROOT / "data/MedRBench/demo_stage1_manifest.json").read_text(encoding="utf-8"))
    d957 = json.loads((ROOT / "data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json").read_text(encoding="utf-8"))
    t496 = json.loads((ROOT / "data/MedRBench/treatment_496_cases_with_rare_disease_165.json").read_text(encoding="utf-8"))

    def rare_pct(ids: list[str], pool: dict) -> int:
        return round(100 * sum(1 for i in ids if pool.get(i, {}).get("checked_rare_disease")) / len(ids))

    if rare_pct(m["diagnosis"]["case_ids"], d957) != 65:
        issues.append(f"diag demo rare % got {rare_pct(m['diagnosis']['case_ids'], d957)} expected 65")
    if rare_pct(list(d957.keys()), d957) != 51:
        issues.append(f"diag full rare % got {rare_pct(list(d957.keys()), d957)} expected 51")
    if rare_pct(m["treatment"]["case_ids"], t496) != 51:
        issues.append(f"treat demo rare % got {rare_pct(m['treatment']['case_ids'], t496)} expected 51")
    if rare_pct(list(t496.keys()), t496) != 33:
        issues.append(f"treat full rare % got {rare_pct(list(t496.keys()), t496)} expected 33")

    # §6.5 scope
    scope_exp = {
        "diagnosis_2b_deepseek-r1_reparsed.json": ("direct", 4.65, "sae_augmented", 4.07),
        "diagnosis_2b_o3-mini.json": ("direct", 4.93, None, None),
        "diagnosis_2b_qwen3-8b_reparsed.json": ("direct", 4.71, "sae_augmented", 4.20),
    }
    for fname, (gd, ed, ga, ea) in scope_exp.items():
        p = SCOPE / fname
        if not p.is_file():
            issues.append(f"missing scope file {fname}")
            continue
        nd, md = scope_mean(p, gd)
        if md is not None and abs(md - ed) > 0.02:
            issues.append(f"scope {fname} direct mean {md} expected {ed}")
        if ga and ea:
            na, ma = scope_mean(p, ga)
            if ma is not None and abs(ma - ea) > 0.02:
                issues.append(f"scope {fname} aug mean {ma} expected {ea}")

    for fname in ["diagnosis_9b_o3-mini.json", "diagnosis_9b_deepseek-r1.json", "diagnosis_9b_qwen3-8b.json"]:
        p = SCOPE / fname
        if p.is_file():
            for g in ["direct", "sae_augmented"]:
                n, mean = scope_mean(p, g)
                if n and mean != 5.0:
                    issues.append(f"9B scope {fname} {g} mean {mean} expected 5.0")

    # §6.4 variance
    by = {}
    for s in SUBJ:
        doc = json.loads((RE / f"diagnosis_gemma-9b-it_{s}_direct.json").read_text(encoding="utf-8"))
        by[s] = {cid: r for cid, r in doc["cases"].items() if r.get("status") == "ok"}
    common = sorted(set.intersection(*[set(by[s]) for s in SUBJ]))
    var_exp = {"efficiency": (52.6, 71.3), "factulity": (50.2, 74.9), "recall": (63.3, 55.5)}
    for k, (eb, ew) in var_exp.items():
        case_means = [st.mean([by[s][c][k] for s in SUBJ]) for c in common]
        between = sum((m - st.mean(case_means)) ** 2 for m in case_means) / len(case_means)
        within = sum(
            (by[s][c][k] - st.mean([by[x][c][k] for x in SUBJ])) ** 2
            for c in common
            for s in SUBJ
        ) / (len(common) * len(SUBJ))
        total = between + within
        bp, wp = round(100 * between / total, 1), round(100 * within / total, 1)
        if abs(bp - eb) > 0.2 or abs(wp - ew) > 0.2:
            issues.append(f"variance {k}: got between={bp}% within={wp}% expected {eb}/{ew}")

    # typical cases
    cases = {
        "PMC11439974": ("deepseek-r1", "recall", 0.50),
        "PMC11470589": ("deepseek-r1", "factulity", 0.40),
        "PMC11418098": ("deepseek-r1", "recall", 0.50),
        "PMC11395317": ("qwen3-8b", "recall", 0.17),
    }
    b2 = {
        s: json.loads((RE / f"diagnosis_gemma-2b-it_{s}_direct.json").read_text(encoding="utf-8"))["cases"]
        for s in SUBJ
    }
    for cid, (subj, key, exp) in cases.items():
        src = by if subj in by and cid in by[subj] else b2
        got = src[subj][cid][key]
        if abs(got - exp) > 0.01:
            issues.append(f"case {cid} {subj} {key}={got} expected {exp}")

    o3_rec = by["o3-mini"]["PMC11418098"]["recall"]
    qwen_rec = by["qwen3-8b"]["PMC11418098"]["recall"]
    if abs(o3_rec - 0.90) > 0.01 or abs(qwen_rec - 0.70) > 0.01:
        issues.append(f"PMC11418098 o3={o3_rec} qwen={qwen_rec} expected 0.90/0.70")

    o3_rec2 = b2["o3-mini"]["PMC11395317"]["recall"]
    if abs(o3_rec2 - 0.67) > 0.01:
        issues.append(f"PMC11395317 o3 rec {o3_rec2} expected 0.67")

    if issues:
        print(f"FAILED: {len(issues)}")
        for i in issues:
            print(" ", i)
        raise SystemExit(1)
    print("OK: extra sections (rare %, scope, variance, cases) match report.")


if __name__ == "__main__":
    main()
