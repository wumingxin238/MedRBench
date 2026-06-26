#!/usr/bin/env python3
"""Verify STAGE2_EXPERIMENT_REPORT.md numeric claims against raw JSON."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ["qwen3-14b-thinking", "o3-mini", "deepseek-r1"]
MET = ["efficiency", "factulity", "recall"]
TOL_PP = 0.15
TOL_R = 0.02


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def mean(vals: list[float]) -> float:
    return st.mean(vals) if vals else float("nan")


def pp(x: float) -> float:
    return round(x * 100, 1)


def delta_pp(a: float, b: float) -> float:
    return round((a - b) * 100, 1)


def check(label: str, got: float, exp: float, tol: float = TOL_PP) -> str | None:
    if abs(got - exp) <= tol:
        return None
    return f"{label}: got {got} expected {exp} (diff {got - exp:+.2f})"


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def load_acc(model: str, source: str) -> dict[str, bool]:
    sub = "acc_results_gpt" if source == "gpt" else "acc_results_gemma"
    d = DATA / f"Stage2/{sub}/{model}"
    return {f.stem: bool(load(f).get("accuracy")) for f in d.glob("*.json")}


def load_re(model: str, group: str) -> dict[str, dict]:
    p = DATA / f"Stage2/reasoning_eval/diagnosis_gemma-9b-it_{model}_{group}.json"
    doc = load(p)
    return {cid: r for cid, r in doc["cases"].items() if r.get("status") == "ok"}


def acc_stats(acc: dict[str, bool], demo: set[str], hard: set[str]) -> dict:
    tot = sum(acc.values())
    demo_ok = sum(1 for c in demo if acc.get(c))
    hard_ok = sum(1 for c in hard if acc.get(c))
    return {
        "n": len(acc),
        "tot": tot,
        "pct400": pp(tot / 400),
        "demo_ok": demo_ok,
        "demo_pct": demo_ok,  # integer %
        "hard_ok": hard_ok,
        "hard_pct": pp(hard_ok / 300),
        "drop": delta_pp(demo_ok / 100, hard_ok / 300),
    }


def re_means(model: str, group: str, ids: set[str] | None = None) -> dict[str, float]:
    cases = load_re(model, group)
    if ids is not None:
        cases = {c: cases[c] for c in ids if c in cases}
    return {k: mean([r[k] for r in cases.values()]) for k in MET}


def main() -> None:
    manifest = load(DATA / "MedRBench/stage2_manifest.json")
    demo = set(manifest["diagnosis"]["demo_case_ids"])
    hard = set(manifest["diagnosis"]["hard_case_ids"])
    all_ids = manifest["diagnosis"]["case_ids"]
    issues: list[str] = []

    issues.append(check("manifest demo", len(demo), 100, 0))
    issues.append(check("manifest hard", len(hard), 300, 0))
    issues.append(check("manifest total", len(all_ids), 400, 0))

    # --- Expected from report ---
    exp_gemma_acc = {
        "deepseek-r1": {"pct400": 84.2, "tot": 337, "demo": 97, "hard_pct": 80.0, "drop": 17.0},
        "qwen3-14b-thinking": {"pct400": 83.8, "tot": 335, "demo": 92, "hard_pct": 81.0, "drop": 11.0},
        "o3-mini": {"pct400": 78.5, "tot": 314, "demo": 91, "hard_pct": 74.3, "drop": 16.7},
    }
    exp_gpt_acc = {
        "deepseek-r1": {"pct400": 81.5, "demo": 96, "hard_pct": 76.7, "delta": 2.8},
        "qwen3-14b-thinking": {"pct400": 77.8, "demo": 91, "hard_pct": 73.3, "delta": 6.0},
        "o3-mini": {"pct400": 77.5, "demo": 92, "hard_pct": 72.7, "delta": 1.0},
    }
    exp_agree = {"deepseek-r1": 91.8, "o3-mini": 90.0, "qwen3-14b-thinking": 88.5}
    exp_direct = {
        "deepseek-r1": (98.5, 92.6, 90.3),
        "o3-mini": (96.3, 95.4, 93.4),
        "qwen3-14b-thinking": (97.8, 93.2, 89.2),
    }
    exp_aug_delta = {
        "deepseek-r1": (-9.3, 1.4),
        "o3-mini": (-5.8, 1.1),
        "qwen3-14b-thinking": (-8.7, 2.4),
    }
    exp_acc_x_re = {
        "qwen3-14b-thinking": (10.4, 0.26),
        "deepseek-r1": (6.1, 0.17),
        "o3-mini": (5.7, 0.19),
    }

    print("=== Gemma Acc ===")
    gemma = {}
    for m in MODELS:
        acc = load_acc(m, "gemma")
        s = acc_stats(acc, demo, hard)
        gemma[m] = (acc, s)
        e = exp_gemma_acc[m]
        issues.append(check(f"gemma acc files {m}", s["n"], 400, 0))
        issues.append(check(f"gemma acc tot {m}", s["tot"], e["tot"], 0))
        issues.append(check(f"gemma acc 400% {m}", s["pct400"], e["pct400"]))
        issues.append(check(f"gemma demo {m}", s["demo_ok"], e["demo"], 0))
        issues.append(check(f"gemma hard% {m}", s["hard_pct"], e["hard_pct"]))
        issues.append(check(f"gemma drop {m}", s["drop"], e["drop"], 0.1))
        print(f"  {m}: {s['tot']}/400={s['pct400']}% demo={s['demo_ok']} hard={s['hard_ok']}/300={s['hard_pct']}%")

    print("\n=== GPT Acc ===")
    for m in MODELS:
        acc_g = gemma[m][0]
        acc_p = load_acc(m, "gpt")
        s = acc_stats(acc_p, demo, hard)
        e = exp_gpt_acc[m]
        issues.append(check(f"gpt acc 400% {m}", s["pct400"], e["pct400"]))
        issues.append(check(f"gpt demo {m}", s["demo_ok"], e["demo"], 0))
        issues.append(check(f"gpt hard% {m}", s["hard_pct"], e["hard_pct"]))
        d = delta_pp(sum(acc_g.values()) / 400, sum(acc_p.values()) / 400)
        issues.append(check(f"gemma-gpt delta {m}", d, e["delta"]))
        ids = set(acc_g) & set(acc_p)
        agree = sum(1 for i in ids if bool(acc_g[i]) == bool(acc_p[i]))
        agree_pct = pp(agree / len(ids))
        issues.append(check(f"judge agree {m}", agree_pct, exp_agree[m]))
        print(f"  {m}: {sum(acc_p.values())}/400={s['pct400']}% delta={d} agree={agree_pct}%")

    print("\n=== Reasoning direct (400) ===")
    for m in MODELS:
        mets = re_means(m, "direct")
        e = exp_direct[m]
        issues.append(check(f"direct Eff {m}", pp(mets["efficiency"]), e[0]))
        issues.append(check(f"direct Fact {m}", pp(mets["factulity"]), e[1]))
        issues.append(check(f"direct Rec {m}", pp(mets["recall"]), e[2]))
        print(f"  {m}: Eff={pp(mets['efficiency'])} Fact={pp(mets['factulity'])} Rec={pp(mets['recall'])}")

    print("\n=== aug - direct ===")
    for m in MODELS:
        d = re_means(m, "direct")
        a = re_means(m, "inference_augmented")
        de = delta_pp(a["efficiency"], d["efficiency"])
        dr = delta_pp(a["recall"], d["recall"])
        e = exp_aug_delta[m]
        issues.append(check(f"aug dEff {m}", de, e[0]))
        issues.append(check(f"aug dRec {m}", dr, e[1]))
        print(f"  {m}: dEff={de} dRec={dr}")

    print("\n=== Acc x reasoning (direct) ===")
    for m in MODELS:
        acc = gemma[m][0]
        cases = load_re(m, "direct")
        correct = [cases[c] for c in all_ids if acc.get(c) and c in cases]
        wrong = [cases[c] for c in all_ids if c in acc and not acc[c] and c in cases]
        rec_d = delta_pp(mean([x["recall"] for x in correct]), mean([x["recall"] for x in wrong]))
        r = pearson(
            [1.0 if acc[c] else 0.0 for c in all_ids if c in cases],
            [cases[c]["recall"] for c in all_ids if c in cases],
        )
        e = exp_acc_x_re[m]
        issues.append(check(f"Rec delta {m}", rec_d, e[0]))
        issues.append(check(f"r acc,Rec {m}", round(r, 2), e[1], TOL_R))
        print(f"  {m}: RecΔ={rec_d} r={round(r,2)}")

    # Derived claims
    print("\n=== Derived claims ===")
    g14 = gemma["qwen3-14b-thinking"][1]
    gds = gemma["deepseek-r1"][1]
    go3 = gemma["o3-mini"][1]
    hard_gap_14_o3 = delta_pp(g14["hard_ok"] / 300, go3["hard_ok"] / 300)
    issues.append(check("14B vs o3 hard gap", hard_gap_14_o3, 6.7))
    acc_gap_14_o3 = delta_pp(g14["pct400"] / 100, go3["pct400"] / 100)
    issues.append(check("14B vs o3 400 gap", acc_gap_14_o3, 5.3))

    # 14B wrong count
    acc14 = gemma["qwen3-14b-thinking"][0]
    wrong14 = sum(1 for c in all_ids if c in acc14 and not acc14[c])
    wrong14_hard = sum(1 for c in hard if c in acc14 and not acc14[c])
    issues.append(check("14B gemma wrong", wrong14, 65, 0))
    issues.append(check("14B wrong from hard %", pp(wrong14_hard / wrong14) if wrong14 else 0, 87.7, 0.5))

    acc14_gpt = load_acc("qwen3-14b-thinking", "gpt")
    wrong14_gpt = sum(1 for c in all_ids if c in acc14_gpt and not acc14_gpt[c])
    issues.append(check("14B gpt wrong", wrong14_gpt, 89, 0))

    # 14B demo vs hard process (direct) — Eff/Rec ≤ 2 pp per report §2.3
    m14_demo = re_means("qwen3-14b-thinking", "direct", demo)
    m14_hard = re_means("qwen3-14b-thinking", "direct", hard)
    for k in ("efficiency", "recall"):
        diff = abs(pp(m14_demo[k]) - pp(m14_hard[k]))
        if diff > 2.0:
            issues.append(f"14B demo-hard {k} diff {diff}pp > 2.0")

    issues.append(check("14B acc demo-hard diff pp", g14["demo_ok"] - g14["hard_pct"], 11.0, 0.5))

    # Stage-1 numbers (from Stage1 acc_results)
    s1_acc = ROOT / "data" / "Stage1" / "acc_results"
    s1_exp = {"deepseek-r1": 95, "o3-mini": 92, "qwen3-8b": 86}
    for subj, exp in s1_exp.items():
        ok = sum(
            1
            for f in (s1_acc / subj).glob("*.json")
            if load(f).get("accuracy")
        )
        issues.append(check(f"stage1 acc {subj}", ok, exp, 0))

    # 14B vs 8B demo Gemma Rec - need stage1 8b reasoning on demo cases
    # Report says 14B vs 8B: +6pp acc (92 vs 86), -7.2pp Rec
    # 14B demo acc already checked; 8b is stage1
    s1_re_8b = load(DATA / "Stage2/reasoning_eval/diagnosis_gemma-9b-it_qwen3-8b_direct.json") if False else None
    # 8b reasoning on stage1 demo - check if exists in Stage1
    p8 = DATA / "Stage1/reasoning_eval/diagnosis_gemma-9b-it_qwen3-8b_direct.json"
    p14 = DATA / "Stage2/reasoning_eval/diagnosis_gemma-9b-it_qwen3-14b-thinking_direct.json"
    if p8.is_file() and p14.is_file():
        r8 = {cid: r for cid, r in load(p8)["cases"].items() if r.get("status") == "ok"}
        r14 = {cid: r for cid, r in load(p14)["cases"].items() if r.get("status") == "ok"}
        common_demo = sorted(demo & set(r8) & set(r14))
        rec8 = mean([r8[c]["recall"] for c in common_demo])
        rec14 = mean([r14[c]["recall"] for c in common_demo])
        issues.append(check("14B vs 8B demo Rec delta", delta_pp(rec14, rec8), -7.2))
        print(f"  14B vs 8B demo Rec: {pp(rec14)} vs {pp(rec8)} delta={delta_pp(rec14, rec8)}")

    # GPT 14B vs o3 "打平" - delta should be ~0.3pp
    p14 = acc_stats(load_acc("qwen3-14b-thinking", "gpt"), demo, hard)
    po3 = acc_stats(load_acc("o3-mini", "gpt"), demo, hard)
    gpt_gap = delta_pp(p14["pct400"] / 100, po3["pct400"] / 100)
    print(f"  GPT 14B vs o3 gap: {gpt_gap} pp")

    # Completeness reasoning
    for m in MODELS:
        for g in ("direct", "inference_augmented"):
            n = len(load_re(m, g))
            if n != 400:
                issues.append(f"reasoning {m} {g}: {n}/400")

    issues = [x for x in issues if x]
    print(f"\n{'=' * 60}")
    if issues:
        print(f"FAILED: {len(issues)} mismatch(es)")
        for i in issues:
            print(f"  - {i}")
        raise SystemExit(1)
    print("OK: all Stage-2 report numbers verified.")


if __name__ == "__main__":
    main()
