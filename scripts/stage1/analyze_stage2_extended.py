#!/usr/bin/env python3
"""Extended stats for STAGE2_EXPERIMENT_REPORT.md."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main() -> None:
    manifest = load(DATA / "MedRBench/stage2_manifest.json")
    demo = set(manifest["diagnosis"]["demo_case_ids"])
    hard = set(manifest["diagnosis"]["hard_case_ids"])
    all_ids = list(manifest["diagnosis"]["case_ids"])

    acc = {
        f.stem: bool(load(f).get("accuracy"))
        for f in (DATA / "Stage2/acc_results_gpt/qwen3-14b-thinking").glob("*.json")
    }
    direct = load(
        DATA / "Stage2/reasoning_eval/diagnosis_gemma-9b-it_qwen3-14b-thinking_direct.json"
    )["cases"]
    aug = load(
        DATA / "Stage2/reasoning_eval/diagnosis_gemma-9b-it_qwen3-14b-thinking_inference_augmented.json"
    )["cases"]

    # Rec spread demo vs hard (same model, between subsets)
    def subset_recs(ids):
        return [direct[c]["recall"] for c in ids if direct.get(c, {}).get("status") == "ok"]

    print("Rec stdev demo:", st.pstdev(subset_recs(demo)))
    print("Rec stdev hard:", st.pstdev(subset_recs(hard)))

    # acc vs rec correlation (direct)
    pairs = [
        (1.0 if acc[c] else 0.0, direct[c]["recall"])
        for c in all_ids
        if c in acc and direct.get(c, {}).get("status") == "ok"
    ]
    print("Pearson(acc, direct Rec):", pearson([p[0] for p in pairs], [p[1] for p in pairs]))

    pairs_aug = [
        (1.0 if acc[c] else 0.0, aug[c]["recall"])
        for c in all_ids
        if c in acc and aug.get(c, {}).get("status") == "ok"
    ]
    print("Pearson(acc, aug Rec):", pearson([p[0] for p in pairs_aug], [p[1] for p in pairs_aug]))

    # wrong cases high rec
    wrong = [c for c in all_ids if c in acc and not acc[c]]
    hi = sorted(
        [(c, direct[c]["recall"], direct[c]["efficiency"], float(direct[c]["factulity"])) for c in wrong if direct.get(c, {}).get("status") == "ok"],
        key=lambda x: -x[1],
    )[:10]
    print("\nWrong + high direct Rec:")
    for row in hi:
        print(f"  {row[0]} Rec={row[1]:.2f} Eff={row[2]:.2f} Fact={row[3]:.2f} hard={row[0] in hard}")

    lo = sorted(
        [(c, direct[c]["recall"]) for c in wrong if direct.get(c, {}).get("status") == "ok"],
        key=lambda x: x[1],
    )[:5]
    print("\nWrong + low direct Rec:")
    for c, r in lo:
        print(f"  {c} Rec={r:.2f}")

    # fact delta
    for group, cases in [("direct", direct), ("aug", aug)]:
        ok = [cases[c] for c in all_ids if acc.get(c) and cases.get(c, {}).get("status") == "ok"]
        bad = [cases[c] for c in all_ids if c in acc and not acc[c] and cases.get(c, {}).get("status") == "ok"]
        df = st.mean([float(x["factulity"]) for x in ok]) - st.mean([float(x["factulity"]) for x in bad])
        print(f"\n{group} Fact delta(c-w): {100*df:+.1f} pp")

    # stage1 qwen8b rec on demo same cases
    s1 = load(DATA / "Stage1/reasoning_eval/diagnosis_gemma-9b-it_qwen3-8b_direct.json")["cases"]
    demo_list = sorted(demo)
    r8 = [s1[c]["recall"] for c in demo_list if s1.get(c, {}).get("status") == "ok"]
    r14 = [direct[c]["recall"] for c in demo_list if direct.get(c, {}).get("status") == "ok"]
    print("\nDemo Rec pearson 8b vs 14b:", pearson(r8, r14))
    print("Demo Rec mean delta 14b-8b:", 100 * (st.mean(r14) - st.mean(r8)), "pp")

    # aug eff spread within 400
    spreads = []
    for c in all_ids:
        d, a = direct.get(c), aug.get(c)
        if d and a and d.get("status") == "ok" and a.get("status") == "ok":
            spreads.append(abs(d["efficiency"] - a["efficiency"]))
    print(f"\n|Eff_direct - Eff_aug| median: {st.median(spreads):.3f}  >=0.2: {sum(1 for s in spreads if s>=0.2)}/400")


if __name__ == "__main__":
    main()
