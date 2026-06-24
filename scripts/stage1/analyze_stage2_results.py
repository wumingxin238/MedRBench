#!/usr/bin/env python3
"""One-off Stage-2 aggregate analysis."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def mean(vals: list[float]) -> float:
    return st.mean(vals) if vals else float("nan")


def main() -> None:
    manifest = load_json(DATA / "MedRBench/stage2_manifest.json")
    demo_ids = set(manifest["diagnosis"]["demo_case_ids"])
    hard_ids = set(manifest["diagnosis"]["hard_case_ids"])
    all_ids = set(manifest["diagnosis"]["case_ids"])

    acc: dict[str, bool] = {}
    acc_dir = DATA / "Stage2/acc_results_gpt/qwen3-14b-thinking"
    for f in acc_dir.glob("*.json"):
        acc[f.stem] = bool(load_json(f).get("accuracy"))

    def load_re(group: str):
        p = DATA / f"Stage2/reasoning_eval/diagnosis_gemma-9b-it_qwen3-14b-thinking_{group}.json"
        d = load_json(p)
        return d["cases"], d.get("meta", {})

    print("=" * 72)
    print("STAGE-2: qwen3-14b-thinking × 400 (100 demo + 300 hard)")
    print("=" * 72)

    total_ok = sum(acc.values())
    demo_ok = sum(1 for cid in demo_ids if acc.get(cid))
    hard_ok = sum(1 for cid in hard_ids if acc.get(cid))
    print("\n## 1. Diagnosis Accuracy (GPT-4o judge)")
    print(f"  Overall:  {total_ok}/400 = {100 * total_ok / 400:.1f}%")
    print(f"  Demo 100: {demo_ok}/100 = {demo_ok:.0f}%")
    print(f"  Hard 300: {hard_ok}/300 = {100 * hard_ok / 300:.1f}%")

    for group in ("direct", "inference_augmented"):
        cases, meta = load_re(group)
        print(f"\n## 2. Gemma-9B reasoning_eval · {group}")
        print(
            f"  All 400: Eff={meta.get('mean_efficiency', 0)*100:.1f}% "
            f"Fact={meta.get('mean_factulity', 0)*100:.1f}% "
            f"Rec={meta.get('mean_recall', 0)*100:.1f}%"
        )
        for label, ids in [("Demo 100", demo_ids), ("Hard 300", hard_ids)]:
            rows = [cases[c] for c in ids if cases.get(c, {}).get("status") == "ok"]
            print(
                f"  {label}: Eff={mean([r['efficiency'] for r in rows])*100:.1f}% "
                f"Fact={mean([float(r['factulity']) for r in rows])*100:.1f}% "
                f"Rec={mean([r['recall'] for r in rows])*100:.1f}% "
                f"| Acc={sum(1 for c in ids if acc.get(c))}/{len(ids)}"
            )

    direct, _ = load_re("direct")
    aug, _ = load_re("inference_augmented")
    de, dr, ae, ar = [], [], [], []
    for cid in all_ids:
        d, a = direct.get(cid), aug.get(cid)
        if d and a and d.get("status") == "ok" and a.get("status") == "ok":
            de.append(d["efficiency"])
            dr.append(d["recall"])
            ae.append(a["efficiency"])
            ar.append(a["recall"])
    print("\n## 3. direct → aug 组间效应 (paired 400)")
    print(f"  Eff: {mean(de)*100:.1f}% → {mean(ae)*100:.1f}%  ({100*(mean(ae)-mean(de)):+.1f} pp)")
    print(f"  Rec: {mean(dr)*100:.1f}% → {mean(ar)*100:.1f}%  ({100*(mean(ar)-mean(dr)):+.1f} pp)")

    for group in ("direct", "inference_augmented"):
        cases, _ = load_re(group)
        correct = [
            cases[c]
            for c in all_ids
            if acc.get(c) and cases.get(c, {}).get("status") == "ok"
        ]
        wrong = [
            cases[c]
            for c in all_ids
            if c in acc and not acc[c] and cases.get(c, {}).get("status") == "ok"
        ]
        print(f"\n## 4. Accuracy × reasoning 分层 · {group}")
        print(
            f"  答对 n={len(correct):3d}: Eff={mean([x['efficiency'] for x in correct])*100:.1f}% "
            f"Fact={mean([float(x['factulity']) for x in correct])*100:.1f}% "
            f"Rec={mean([x['recall'] for x in correct])*100:.1f}%"
        )
        print(
            f"  答错 n={len(wrong):3d}: Eff={mean([x['efficiency'] for x in wrong])*100:.1f}% "
            f"Fact={mean([float(x['factulity']) for x in wrong])*100:.1f}% "
            f"Rec={mean([x['recall'] for x in wrong])*100:.1f}%"
        )
        if correct and wrong:
            dc = mean([x["recall"] for x in correct]) - mean([x["recall"] for x in wrong])
            de2 = mean([x["efficiency"] for x in correct]) - mean([x["efficiency"] for x in wrong])
            print(f"  delta(correct-wrong): Rec {100*dc:+.1f} pp, Eff {100*de2:+.1f} pp")

    # Stage-1 demo compare
    s1 = DATA / "Stage1/acc_results"
    if s1.is_dir():
        print("\n## 5. Demo 100 与 Stage-1 对比 (accuracy)")
        for model in ["o3-mini", "deepseek-r1", "qwen3-8b", "qwen3-14b-thinking"]:
            if model == "qwen3-14b-thinking":
                ok = demo_ok
            else:
                d = s1 / model
                ok = sum(
                    1
                    for cid in demo_ids
                    if (d / f"{cid}.json").is_file()
                    and load_json(d / f"{cid}.json").get("accuracy")
                )
            print(f"  {model:22s} {ok}/100")

    # Stage-1 gemma9b direct on demo for qwen3-8b
    s1_re = DATA / "Stage1/reasoning_eval/diagnosis_gemma-9b-it_qwen3-8b_direct.json"
    s2_re = DATA / "Stage2/reasoning_eval/diagnosis_gemma-9b-it_qwen3-14b-thinking_direct.json"
    if s1_re.is_file() and s2_re.is_file():
        s1c = load_json(s1_re)["cases"]
        s2c = load_json(s2_re)["cases"]
        print("\n## 6. Demo 100 Gemma-9B direct 对比 (qwen3-8b Stage1 vs qwen3-14b-thinking Stage2)")
        for label, src, ids in [
            ("qwen3-8b (S1)", s1c, demo_ids),
            ("qwen3-14b-thinking (S2 demo)", s2c, demo_ids),
            ("qwen3-14b-thinking (S2 hard)", s2c, hard_ids),
        ]:
            rows = [src[c] for c in ids if src.get(c, {}).get("status") == "ok"]
            print(
                f"  {label:30s} Eff={mean([r['efficiency'] for r in rows])*100:.1f}% "
                f"Fact={mean([float(r['factulity']) for r in rows])*100:.1f}% "
                f"Rec={mean([r['recall'] for r in rows])*100:.1f}%"
            )

    # error case patterns
    wrong_ids = sorted(c for c in all_ids if c in acc and not acc[c])
    print(f"\n## 7. 错例概况 (n={len(wrong_ids)})")
    hard_wrong = sum(1 for c in wrong_ids if c in hard_ids)
    print(f"  Hard 占错例: {hard_wrong}/{len(wrong_ids)} ({100*hard_wrong/len(wrong_ids):.0f}%)")
    low_rec_wrong = sum(1 for c in wrong_ids if direct.get(c, {}).get("recall", 1) < 0.5)
    print(f"  direct Rec<0.5 的错例: {low_rec_wrong}/{len(wrong_ids)}")

    cases400 = load_json(DATA / "MedRBench/diagnosis_400.json")

    def is_rare(cid: str) -> bool:
        return bool(cases400.get(cid, {}).get("checked_rare_disease"))

    rare_ids = [c for c in acc if is_rare(c)]
    non_rare = [c for c in acc if not is_rare(c)]
    print("\n## 8. Rare disease 分层 (accuracy)")
    print(
        f"  Rare ({len(rare_ids)}): {sum(acc[c] for c in rare_ids)}/{len(rare_ids)} "
        f"= {100*mean([float(acc[c]) for c in rare_ids]):.1f}%"
    )
    print(
        f"  Non-rare ({len(non_rare)}): {sum(acc[c] for c in non_rare)}/{len(non_rare)} "
        f"= {100*mean([float(acc[c]) for c in non_rare]):.1f}%"
    )


if __name__ == "__main__":
    main()
