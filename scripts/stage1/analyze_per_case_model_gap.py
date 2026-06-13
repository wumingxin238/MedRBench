#!/usr/bin/env python3
"""Per-case score spread: o3-mini vs deepseek-r1 vs qwen3-8b (Gemma Judge reasoning_eval)."""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data" / "Stage1" / "reasoning_eval"

SUBJECTS = ["o3-mini", "deepseek-r1", "qwen3-8b"]
METRICS = ["efficiency", "factulity", "recall"]
METRIC_LABEL = {"efficiency": "Efficiency", "factulity": "Factuality", "recall": "Completeness"}


def load_scores(judge: str, group: str, subject: str) -> dict[str, dict[str, float]]:
    path = DEFAULT_DIR / f"diagnosis_{judge}_{subject}_{group}.json"
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {}
    for cid, row in doc.get("cases", {}).items():
        if row.get("status") != "ok":
            continue
        out[cid] = {m: float(row[m]) for m in METRICS if row.get(m) is not None}
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def analyze(judge: str, group: str) -> None:
    by_subj = {s: load_scores(judge, group, s) for s in SUBJECTS}
    common = set.intersection(*(set(d) for d in by_subj.values() if d))
    if not common:
        print(f"No overlapping ok cases for judge={judge} group={group}")
        return

    print(f"\n{'=' * 72}")
    print(f"Judge: {judge}  Group: {group}  Cases: {len(common)}/100")
    print(f"{'=' * 72}")

    # --- aggregate means (replicate report) ---
    print("\n[Aggregate means]")
    for s in SUBJECTS:
        rows = [by_subj[s][c] for c in common if c in by_subj[s]]
        if not rows:
            continue
        print(
            f"  {s:14} "
            f"eff={st.mean(r['efficiency'] for r in rows):.3f} "
            f"fact={st.mean(r['factulity'] for r in rows):.3f} "
            f"rec={st.mean(r['recall'] for r in rows):.3f}"
        )

    # --- per-case spread across 3 models ---
    print("\n[Per-case spread across 3 models (max-min on each case)]")
    spreads: dict[str, list[float]] = {m: [] for m in METRICS}
    case_spread: dict[str, dict[str, float]] = {}

    for cid in sorted(common):
        case_spread[cid] = {}
        for m in METRICS:
            vals = [by_subj[s][cid][m] for s in SUBJECTS]
            rng = max(vals) - min(vals)
            spreads[m].append(rng)
            case_spread[cid][m] = rng

    for m in METRICS:
        sp = spreads[m]
        print(
            f"  {METRIC_LABEL[m]:14} range: mean={st.mean(sp):.4f}  "
            f"median={st.median(sp):.4f}  p90={sorted(sp)[int(0.9 * len(sp)) - 1]:.4f}  "
            f"max={max(sp):.4f}  "
            f"cases with range=0: {sum(1 for x in sp if x == 0)}/{len(sp)}  "
            f"range>=0.2: {sum(1 for x in sp if x >= 0.2)}/{len(sp)}"
        )

    # --- variance decomposition (approx) ---
    print("\n[Variance: case difficulty vs model gap]")
    for m in METRICS:
        matrix = {s: [by_subj[s][cid][m] for cid in sorted(common)] for s in SUBJECTS}
        all_vals = [v for s in SUBJECTS for v in matrix[s]]
        grand_mean = st.mean(all_vals)
        # between-case variance: variance of case means
        case_means = [st.mean([matrix[s][i] for s in SUBJECTS]) for i in range(len(common))]
        var_case = st.variance(case_means) if len(case_means) > 1 else 0.0
        # within-case variance: mean of per-case variance across 3 models
        within = []
        for i, cid in enumerate(sorted(common)):
            vals = [matrix[s][i] for s in SUBJECTS]
            if len(set(vals)) > 1:
                within.append(st.variance(vals))
            else:
                within.append(0.0)
        var_within = st.mean(within)
        total = st.variance(all_vals) if len(all_vals) > 1 else 0.0
        print(
            f"  {METRIC_LABEL[m]:14} total_var={total:.5f}  "
            f"between_case={var_case:.5f} ({100 * var_case / total if total else 0:.1f}%)  "
            f"within_case(model)={var_within:.5f} ({100 * var_within / total if total else 0:.1f}%)"
        )

    # --- pairwise correlation (cases as units) ---
    print("\n[Pairwise Pearson r across cases (same metric)]")
    pairs = [("o3-mini", "deepseek-r1"), ("o3-mini", "qwen3-8b"), ("deepseek-r1", "qwen3-8b")]
    for m in METRICS:
        rs = []
        for a, b in pairs:
            xs = [by_subj[a][cid][m] for cid in sorted(common)]
            ys = [by_subj[b][cid][m] for cid in sorted(common)]
            r = pearson(xs, ys)
            rs.append(f"{a[:3]}-{b[:3]}={r:.3f}")
        print(f"  {METRIC_LABEL[m]:14} " + "  ".join(rs))

    # --- model pairwise mean abs diff per case ---
    print("\n[Mean |delta| per case between model pairs]")
    for m in METRICS:
        for a, b in pairs:
            diffs = [abs(by_subj[a][cid][m] - by_subj[b][cid][m]) for cid in sorted(common)]
            print(
                f"  {METRIC_LABEL[m]:14} {a:14} vs {b:14}  "
                f"mean|d|={st.mean(diffs):.4f}  max|d|={max(diffs):.4f}"
            )

    # --- top spread cases ---
    print("\n[Top 5 cases by total spread (sum of 3 metric ranges)]")
    ranked = sorted(
        common,
        key=lambda c: sum(case_spread[c][m] for m in METRICS),
        reverse=True,
    )
    for cid in ranked[:5]:
        parts = "  ".join(f"{METRIC_LABEL[m][:4]}={case_spread[cid][m]:.2f}" for m in METRICS)
        vals = " | ".join(
            f"{s[:6]}:e{by_subj[s][cid]['efficiency']:.2f}f{by_subj[s][cid]['factulity']:.2f}r{by_subj[s][cid]['recall']:.2f}"
            for s in SUBJECTS
        )
        print(f"  {cid}  spread({parts})  [{vals}]")

    # --- cases where qwen clearly worse than o3 ---
    print("\n[Cases where qwen recall < o3 recall by >= 0.2] (weak vs strong signal)")
    big = [
        cid
        for cid in sorted(common)
        if by_subj["o3-mini"][cid]["recall"] - by_subj["qwen3-8b"][cid]["recall"] >= 0.2
    ]
    print(f"  count={len(big)}/{len(common)}")
    for cid in big[:8]:
        o, q = by_subj["o3-mini"][cid]["recall"], by_subj["qwen3-8b"][cid]["recall"]
        print(f"    {cid}  o3={o:.2f} qwen={q:.2f} delta={o - q:.2f}")

    print("\n[Cases where qwen recall > o3 recall by >= 0.2] (counter-examples)")
    big2 = [
        cid
        for cid in sorted(common)
        if by_subj["qwen3-8b"][cid]["recall"] - by_subj["o3-mini"][cid]["recall"] >= 0.2
    ]
    print(f"  count={len(big2)}/{len(common)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="gemma-2b-it")
    parser.add_argument("--group", default="direct", choices=["direct", "inference_augmented"])
    parser.add_argument("--also-9b", action="store_true", help="Also run for qwen 9b if files exist")
    args = parser.parse_args()

    analyze(args.judge, args.group)
    if args.also_9b:
        # 9b only has qwen locally; skip cross-model for 9b unless all three exist
        for g in ["direct", "inference_augmented"]:
            p = DEFAULT_DIR / f"diagnosis_gemma-9b-it_qwen3-8b_{g}.json"
            if p.is_file():
                doc = json.loads(p.read_text(encoding="utf-8"))
                ok = sum(1 for c in doc["cases"].values() if c.get("status") == "ok")
                print(f"\n(note) gemma-9b-it qwen3-8b {g}: {ok}/100 ok — no 9b o3/deepseek for 3-way compare")


if __name__ == "__main__":
    main()
