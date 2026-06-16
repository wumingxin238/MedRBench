#!/usr/bin/env python3
"""Analyze Gemma-9B reasoning_eval (including partial o3/deepseek runs)."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RE = PROJECT_ROOT / "data" / "Stage1" / "reasoning_eval"
MET = ["efficiency", "factulity", "recall"]
LAB = ["Eff", "Fact", "Rec"]
SUBJ = ["o3-mini", "deepseek-r1", "qwen3-8b"]


def load(judge: str, subj: str, group: str) -> dict[str, dict]:
    p = RE / f"diagnosis_{judge}_{subj}_{group}.json"
    if not p.is_file():
        return {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {cid: row for cid, row in doc["cases"].items() if row.get("status") == "ok"}


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    print("=" * 70)
    print("9B file coverage")
    print("=" * 70)
    for p in sorted(RE.glob("diagnosis_gemma-9b-it_*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        ok = sum(1 for c in doc["cases"].values() if c.get("status") == "ok")
        err = sum(1 for c in doc["cases"].values() if c.get("status") == "error")
        print(f"  {p.name}: {ok}/100 ok, {err} err")

    print("\n" + "=" * 70)
    print("9B direct — 3-model (ok cases)")
    print("=" * 70)
    by9 = {s: load("gemma-9b-it", s, "direct") for s in SUBJ}
    common9 = set.intersection(*[set(by9[s]) for s in SUBJ])
    print(f"  All 3 models ok on same cases: {len(common9)}/100")
    for s in SUBJ:
        rows = list(by9[s].values())
        print(f"  {s:14} n_ok={len(rows)}", end="")
        if rows:
            print(
                f"  eff={st.mean(r['efficiency'] for r in rows):.3f} "
                f"fact={st.mean(r['factulity'] for r in rows):.3f} "
                f"rec={st.mean(r['recall'] for r in rows):.3f}"
            )
        else:
            print()

    if len(common9) >= 3:
        print(f"\n  [{len(common9)}-case intersection] 3-model means")
        for s in SUBJ:
            rows = [by9[s][c] for c in sorted(common9)]
            print(
                f"    {s:14} eff={st.mean(r['efficiency'] for r in rows):.3f} "
                f"fact={st.mean(r['factulity'] for r in rows):.3f} "
                f"rec={st.mean(r['recall'] for r in rows):.3f}"
            )
        print("\n  Per-case spread (max-min across 3 models)")
        pairs = [
            ("o3-mini", "deepseek-r1"),
            ("o3-mini", "qwen3-8b"),
            ("deepseek-r1", "qwen3-8b"),
        ]
        for k, label in zip(MET, LAB):
            sp = [
                max(by9[s][c][k] for s in SUBJ) - min(by9[s][c][k] for s in SUBJ)
                for c in sorted(common9)
            ]
            print(
                f"    {label}: mean={st.mean(sp):.3f} median={st.median(sp):.3f} "
                f"max={max(sp):.3f} range>=0.2: {sum(1 for x in sp if x >= 0.2)}/{len(sp)}"
            )
        print("\n  Mean |delta| between model pairs")
        for k, label in zip(MET, LAB):
            for a, b in pairs:
                diffs = [abs(by9[a][c][k] - by9[b][c][k]) for c in sorted(common9)]
                print(
                    f"    {label} {a:14} vs {b:14} mean|d|={st.mean(diffs):.3f} max={max(diffs):.3f}"
                )

    print("\n" + "=" * 70)
    print("9B vs 2B (direct) on overlapping ok cases")
    print("=" * 70)
    for s in SUBJ:
        b9, b2 = by9[s], load("gemma-2b-it", s, "direct")
        common = sorted(set(b9) & set(b2))
        if not common:
            print(f"  {s}: no overlap")
            continue
        print(f"  {s} (n={len(common)}):")
        for k, label in zip(MET, LAB):
            v9 = [b9[c][k] for c in common]
            v2 = [b2[c][k] for c in common]
            print(
                f"    {label}: 9B={st.mean(v9):.3f} 2B={st.mean(v2):.3f} "
                f"delta={(st.mean(v9)-st.mean(v2))*100:+.1f}pp  r={pearson(v9,v2):.3f}"
            )

    print("\n" + "=" * 70)
    print("9B qwen3-8b: direct vs aug (100/100)")
    print("=" * 70)
    q_d = load("gemma-9b-it", "qwen3-8b", "direct")
    q_a = load("gemma-9b-it", "qwen3-8b", "inference_augmented")
    for k, label in zip(MET, LAB):
        m0 = st.mean(q_d[c][k] for c in q_d)
        m1 = st.mean(q_a[c][k] for c in q_a)
        print(f"  {label}: direct={m0:.3f}  aug={m1:.3f}  ({(m1-m0)*100:+.1f} pp)")

    print("\n" + "=" * 70)
    print("9B qwen3-8b distribution (direct / aug)")
    print("=" * 70)
    for name, rows in [("direct", list(q_d.values())), ("aug", list(q_a.values()))]:
        print(f"  [{name}]")
        for k, label in zip(MET, LAB):
            vals = sorted(r[k] for r in rows)
            n = len(vals)
            perfect = sum(1 for v in vals if v >= 0.999)
            low = sum(1 for v in vals if v < 0.9)
            p25 = vals[n // 4]
            p75 = vals[(3 * n) // 4]
            print(
                f"    {label}: min={min(vals):.2f} p25={p25:.2f} med={st.median(vals):.2f} "
                f"p75={p75:.2f} max={max(vals):.2f} perfect={perfect}/{n} low<0.9={low}"
            )
        triple = sum(1 for r in rows if all(r[k] >= 0.999 for k in MET))
        print(f"    triple>=0.999: {triple}/{len(rows)}")

    print("\n" + "=" * 70)
    print("9B direct vs aug (all 3 models)")
    print("=" * 70)
    for s in SUBJ:
        d0 = load("gemma-9b-it", s, "direct")
        d1 = load("gemma-9b-it", s, "inference_augmented")
        print(f"  {s}:")
        for k, label in zip(MET, LAB):
            m0 = st.mean(d0[c][k] for c in d0)
            m1 = st.mean(d1[c][k] for c in d1)
            print(f"    {label}: {m0:.3f} -> {m1:.3f}  ({(m1-m0)*100:+.1f} pp)")

    if len(common9) >= 3:
        print("\n" + "=" * 70)
        print("Rec pairwise (direct): qwen vs strong models")
        print("=" * 70)
        diff_qd = [by9["qwen3-8b"][c]["recall"] - by9["deepseek-r1"][c]["recall"] for c in sorted(common9)]
        diff_qo = [by9["qwen3-8b"][c]["recall"] - by9["o3-mini"][c]["recall"] for c in sorted(common9)]
        diff_od = [by9["o3-mini"][c]["recall"] - by9["deepseek-r1"][c]["recall"] for c in sorted(common9)]
        print(
            f"  qwen - deepseek: mean={st.mean(diff_qd)*100:+.1f}pp  "
            f"qwen wins (>{0}): {sum(1 for x in diff_qd if x > 0)}/100"
        )
        print(
            f"  qwen - o3:       mean={st.mean(diff_qo)*100:+.1f}pp  "
            f"qwen wins: {sum(1 for x in diff_qo if x > 0)}/100"
        )
        print(
            f"  o3 - deepseek:   mean={st.mean(diff_od)*100:+.1f}pp  "
            f"o3 wins: {sum(1 for x in diff_od if x > 0)}/100"
        )

        print("\n" + "=" * 70)
        print(f"{len(common9)}-case Rec winners/losers")
        print("=" * 70)
        for role, fn in [("lowest", min), ("highest", max)]:
            counts = {s: 0 for s in SUBJ}
            for c in sorted(common9):
                scores = {s: by9[s][c]["recall"] for s in SUBJ}
                target = fn(scores.values())
                for s in SUBJ:
                    if scores[s] == target:
                        counts[s] += 1
            print(f"  {role} Rec: " + ", ".join(f"{s}={counts[s]}" for s in SUBJ))

        print("\n  High-spread cases (any metric spread >= 0.2)")
        for c in sorted(common9):
            spreads = {
                label: max(by9[s][c][k] for s in SUBJ) - min(by9[s][c][k] for s in SUBJ)
                for k, label in zip(MET, LAB)
            }
            if max(spreads.values()) >= 0.2:
                sc = {
                    s: tuple(round(by9[s][c][k], 2) for k in MET)
                    for s in SUBJ
                }
                print(f"    {c}: {spreads}  {sc}")


if __name__ == "__main__":
    main()
