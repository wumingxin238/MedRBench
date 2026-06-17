#!/usr/bin/env python3
"""Verify STAGE1_EXPERIMENT_REPORT.md numeric claims against raw JSON."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RE = PROJECT_ROOT / "data" / "Stage1" / "reasoning_eval"
ACC = PROJECT_ROOT / "data" / "Stage1" / "acc_results"
SCOPE = PROJECT_ROOT / "data" / "Stage1" / "gemma_scope"
MANIFEST = PROJECT_ROOT / "data" / "MedRBench" / "demo_stage1_manifest.json"
MET = ["efficiency", "factulity", "recall"]
SUBJ = ["o3-mini", "deepseek-r1", "qwen3-8b"]
TOL_PP = 0.15  # allow 0.15 pp rounding


def load_re(judge: str, subj: str, group: str) -> dict[str, dict]:
    p = RE / f"diagnosis_{judge}_{subj}_{group}.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {cid: r for cid, r in doc["cases"].items() if r.get("status") == "ok"}


def mean(rows: list[dict], k: str) -> float:
    return st.mean(r[k] for r in rows) if rows else float("nan")


def pp(x: float) -> float:
    return round(x * 100, 1)


def delta_pp(a: float, b: float) -> float:
    return round((a - b) * 100, 1)


def check(label: str, got: float, exp: float, tol: float = TOL_PP) -> str | None:
    if abs(got - exp) <= tol:
        return None
    return f"MISMATCH {label}: got {got:.2f} expected {exp:.2f} (diff {got-exp:+.2f})"


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def spread_stats(judge: str, group: str) -> dict:
    by = {s: load_re(judge, s, group) for s in SUBJ}
    common = sorted(set.intersection(*[set(by[s]) for s in SUBJ]))
    out = {}
    for k in MET:
        spreads = [max(by[s][c][k] for s in SUBJ) - min(by[s][c][k] for s in SUBJ) for c in common]
        out[k] = {"median": round(st.median(spreads), 3), "ge02": sum(1 for x in spreads if x >= 0.2)}
    return out


def acc_counts(subj: str) -> tuple[int, int]:
    d = ACC / subj
    ok = err = 0
    for f in d.glob("*.json"):
        if json.loads(f.read_text(encoding="utf-8")).get("accuracy"):
            ok += 1
        else:
            err += 1
    return ok, err


def stratified(judge: str, group: str, subj: str) -> dict:
    rows = load_re(judge, subj, group)
    acc = {f.stem: json.loads(f.read_text(encoding="utf-8"))["accuracy"] for f in (ACC / subj).glob("*.json")}
    common = sorted(set(rows) & set(acc))
    c = [rows[x] for x in common if acc[x]]
    w = [rows[x] for x in common if not acc[x]]
    return {
        "n_c": len(c),
        "n_w": len(w),
        **{f"{k}_c": pp(mean(c, k)) for k in MET},
        **{f"{k}_w": pp(mean(w, k)) for k in MET},
        **{f"{k}_all": pp(mean([rows[x] for x in common], k)) for k in MET},
        **{f"{k}_d": delta_pp(mean(c, k), mean(w, k)) for k in MET},
    }


def main() -> None:
    issues: list[str] = []

    # --- §2 demo manifest ---
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    issues.append(check("diag demo count", len(m["diagnosis"]["case_ids"]), 100, 0))
    issues.append(check("treat demo count", len(m["treatment"]["case_ids"]), 100, 0))

    # --- §5 2B aggregates ---
    exp_2b = {
        ("o3-mini", "direct"): (99.3, 93.4, 78.5),
        ("o3-mini", "inference_augmented"): (98.9, 92.9, 79.7),
        ("deepseek-r1", "direct"): (99.7, 92.4, 78.2),
        ("deepseek-r1", "inference_augmented"): (99.6, 92.1, 80.7),
        ("qwen3-8b", "direct"): (99.6, 92.2, 78.8),
        ("qwen3-8b", "inference_augmented"): (99.1, 92.2, 80.7),
    }
    for (s, g), (e, f, r) in exp_2b.items():
        rows = list(load_re("gemma-2b-it", s, g).values())
        issues.append(check(f"2B {s} {g} Eff", pp(mean(rows, "efficiency")), e))
        issues.append(check(f"2B {s} {g} Fact", pp(mean(rows, "factulity")), f))
        issues.append(check(f"2B {s} {g} Rec", pp(mean(rows, "recall")), r))

    # 2B spread direct
    sp2 = spread_stats("gemma-2b-it", "direct")
    issues.append(check("2B Eff spread med", sp2["efficiency"]["median"], 0.000, 0.001))
    issues.append(check("2B Eff spread>=0.2", sp2["efficiency"]["ge02"], 3, 0))
    issues.append(check("2B Fact spread med", sp2["factulity"]["median"], 0.143, 0.001))
    issues.append(check("2B Fact spread>=0.2", sp2["factulity"]["ge02"], 22, 0))
    issues.append(check("2B Rec spread med", sp2["recall"]["median"], 0.154, 0.001))
    issues.append(check("2B Rec spread>=0.2", sp2["recall"]["ge02"], 39, 0))

    # 2B pearson
    b2 = {s: load_re("gemma-2b-it", s, "direct") for s in SUBJ}
    common = sorted(set.intersection(*[set(b2[s]) for s in SUBJ]))
    pairs = [("o3-deepseek", "o3-mini", "deepseek-r1", 0.476), ("o3-qwen", "o3-mini", "qwen3-8b", 0.424), ("ds-qwen", "deepseek-r1", "qwen3-8b", 0.425)]
    for name, a, b, exp_r in pairs:
        r = pearson([b2[a][c]["recall"] for c in common], [b2[b][c]["recall"] for c in common])
        issues.append(check(f"2B r {name}", round(r, 3), exp_r, 0.01))

    # qwen vs o3 2B
    qlow = sum(1 for c in common if b2["qwen3-8b"][c]["recall"] - b2["o3-mini"][c]["recall"] <= -0.2)
    qhigh = sum(1 for c in common if b2["qwen3-8b"][c]["recall"] - b2["o3-mini"][c]["recall"] >= 0.2)
    issues.append(check("2B qwen<o3 rec>=0.2", qlow, 13, 0))
    issues.append(check("2B qwen>o3 rec>=0.2", qhigh, 11, 0))

    # --- §6.1 9B direct ---
    exp_9d = {
        "o3-mini": (96.4, 96.0, 94.8),
        "deepseek-r1": (98.3, 95.5, 91.5),
        "qwen3-8b": (97.6, 95.4, 96.4),
    }
    b9d = {}
    for s, (e, f, r) in exp_9d.items():
        rows = list(load_re("gemma-9b-it", s, "direct").values())
        b9d[s] = load_re("gemma-9b-it", s, "direct")
        issues.append(check(f"9B direct {s} Eff", pp(mean(rows, "efficiency")), e))
        issues.append(check(f"9B direct {s} Fact", pp(mean(rows, "factulity")), f))
        issues.append(check(f"9B direct {s} Rec", pp(mean(rows, "recall")), r))

    common9 = sorted(set.intersection(*[set(b9d[s]) for s in SUBJ]))
    issues.append(check("qwen-ds rec diff pp", delta_pp(mean([b9d["qwen3-8b"][c] for c in common9], "recall"), mean([b9d["deepseek-r1"][c] for c in common9], "recall")), 4.9))
    issues.append(check("o3-ds rec diff pp", delta_pp(mean([b9d["o3-mini"][c] for c in common9], "recall"), mean([b9d["deepseek-r1"][c] for c in common9], "recall")), 3.3))
    issues.append(check("qwen-o3 rec diff pp", delta_pp(mean([b9d["qwen3-8b"][c] for c in common9], "recall"), mean([b9d["o3-mini"][c] for c in common9], "recall")), 1.7))
    issues.append(check("qwen>ds count", sum(1 for c in common9 if b9d["qwen3-8b"][c]["recall"] > b9d["deepseek-r1"][c]["recall"]), 34, 0))
    issues.append(check("o3>ds count", sum(1 for c in common9 if b9d["o3-mini"][c]["recall"] > b9d["deepseek-r1"][c]["recall"]), 28, 0))
    issues.append(check("qwen>o3 count", sum(1 for c in common9 if b9d["qwen3-8b"][c]["recall"] > b9d["o3-mini"][c]["recall"]), 17, 0))

    def rec_rank(role: str) -> dict[str, int]:
        fn = min if role == "low" else max
        counts = {s: 0 for s in SUBJ}
        for c in common9:
            sc = {s: b9d[s][c]["recall"] for s in SUBJ}
            t = fn(sc.values())
            for s in SUBJ:
                if sc[s] == t:
                    counts[s] += 1
        return counts

    low = rec_rank("low")
    high = rec_rank("high")
    issues.append(check("rec lowest o3", low["o3-mini"], 69, 0))
    issues.append(check("rec lowest ds", low["deepseek-r1"], 83, 0))
    issues.append(check("rec lowest qwen", low["qwen3-8b"], 56, 0))
    issues.append(check("rec highest o3", high["o3-mini"], 81, 0))
    issues.append(check("rec highest ds", high["deepseek-r1"], 65, 0))
    issues.append(check("rec highest qwen", high["qwen3-8b"], 91, 0))

    # --- §6.2 aug ---
    exp_9a = {"o3-mini": 89.2, "deepseek-r1": 87.4, "qwen3-8b": 89.9}
    for s, exp_eff in exp_9a.items():
        rows = list(load_re("gemma-9b-it", s, "inference_augmented").values())
        issues.append(check(f"9B aug {s} Eff", pp(mean(rows, "efficiency")), exp_eff))

    aug_delta = {
        "o3-mini": (-7.3, -0.9, 1.6),
        "deepseek-r1": (-10.9, -1.2, 1.7),
        "qwen3-8b": (-7.6, -1.0, -0.3),
    }
    for s, (de, df, dr) in aug_delta.items():
        d = load_re("gemma-9b-it", s, "direct")
        a = load_re("gemma-9b-it", s, "inference_augmented")
        c = sorted(set(d) & set(a))
        issues.append(check(f"aug delta Eff {s}", delta_pp(mean([a[x] for x in c], "efficiency"), mean([d[x] for x in c], "efficiency")), de))
        issues.append(check(f"aug delta Fact {s}", delta_pp(mean([a[x] for x in c], "factulity"), mean([d[x] for x in c], "factulity")), df))
        issues.append(check(f"aug delta Rec {s}", delta_pp(mean([a[x] for x in c], "recall"), mean([d[x] for x in c], "recall")), dr))

    # --- §6.3 accuracy ---
    acc_exp = {"o3-mini": (92, 8), "deepseek-r1": (95, 5), "qwen3-8b": (86, 14)}
    for s, (ok, bad) in acc_exp.items():
        got_ok, got_bad = acc_counts(s)
        issues.append(check(f"acc ok {s}", got_ok, ok, 0))
        issues.append(check(f"acc err {s}", got_bad, bad, 0))

    # stratified tables
    strat_exp = {
        ("direct", "o3-mini"): {"n_c": 92, "n_w": 8, "efficiency_d": -1.9, "factulity_d": 2.2, "recall_d": 6.8},
        ("direct", "deepseek-r1"): {"n_c": 95, "n_w": 5, "efficiency_d": 2.4, "factulity_d": 3.7, "recall_d": 11.6},
        ("direct", "qwen3-8b"): {"n_c": 86, "n_w": 14, "efficiency_d": -1.2, "factulity_d": -1.6, "recall_d": 0.1},
        ("inference_augmented", "o3-mini"): {"n_c": 92, "n_w": 8, "efficiency_d": 0.4, "factulity_d": 2.7, "recall_d": 9.2},
        ("inference_augmented", "deepseek-r1"): {"n_c": 95, "n_w": 5, "efficiency_d": 4.3, "factulity_d": 4.5, "recall_d": 5.4},
        ("inference_augmented", "qwen3-8b"): {"n_c": 86, "n_w": 14, "efficiency_d": 0.8, "factulity_d": -1.8, "recall_d": 1.3},
    }
    for (g, s), exp in strat_exp.items():
        got = stratified("gemma-9b-it", g, s)
        issues.append(check(f"strat {g} {s} n_c", got["n_c"], exp["n_c"], 0))
        issues.append(check(f"strat {g} {s} n_w", got["n_w"], exp["n_w"], 0))
        for k in MET:
            issues.append(check(f"strat {g} {s} {k}_d", got[f"{k}_d"], exp[f"{k}_d"]))

    # --- §6.4 spreads ---
    sp9d = spread_stats("gemma-9b-it", "direct")
    sp9a = spread_stats("gemma-9b-it", "inference_augmented")
    issues.append(check("9B d Eff spread>=0.2", sp9d["efficiency"]["ge02"], 9, 0))
    issues.append(check("9B a Eff spread med", sp9a["efficiency"]["median"], 0.143, 0.001))
    issues.append(check("9B a Eff spread>=0.2", sp9a["efficiency"]["ge02"], 18, 0))
    issues.append(check("9B d Rec spread med", sp9d["recall"]["median"], 0.050, 0.001))
    issues.append(check("9B d Rec spread>=0.2", sp9d["recall"]["ge02"], 10, 0))
    issues.append(check("9B a Rec spread>=0.2", sp9a["recall"]["ge02"], 11, 0))

    b9 = {s: load_re("gemma-9b-it", s, "direct") for s in SUBJ}
    c9 = sorted(set.intersection(*[set(b9[s]) for s in SUBJ]))
    issues.append(check("9B d qwen<o3", sum(1 for c in c9 if b9["qwen3-8b"][c]["recall"] - b9["o3-mini"][c]["recall"] <= -0.2), 1, 0))
    issues.append(check("9B d qwen>o3", sum(1 for c in c9 if b9["qwen3-8b"][c]["recall"] - b9["o3-mini"][c]["recall"] >= 0.2), 2, 0))
    b9a = {s: load_re("gemma-9b-it", s, "inference_augmented") for s in SUBJ}
    c9a = sorted(set.intersection(*[set(b9a[s]) for s in SUBJ]))
    issues.append(check("9B a qwen<o3", sum(1 for c in c9a if b9a["qwen3-8b"][c]["recall"] - b9a["o3-mini"][c]["recall"] <= -0.2), 2, 0))
    issues.append(check("9B a qwen>o3", sum(1 for c in c9a if b9a["qwen3-8b"][c]["recall"] - b9a["o3-mini"][c]["recall"] >= 0.2), 3, 0))

    # 9B vs 2B judge deltas direct
    b2d = {s: load_re("gemma-2b-it", s, "direct") for s in SUBJ}
    judge_delta = {
        "o3-mini": (-2.8, 2.6, 16.3, 0.35),
        "deepseek-r1": (-1.4, 3.1, 13.3, 0.43),
        "qwen3-8b": (-2.1, 3.2, 17.6, 0.07),
    }
    for s, (de, df, dr, er) in judge_delta.items():
        c = sorted(set(b2d[s]) & set(b9d[s]))
        issues.append(check(f"judge dEff {s}", delta_pp(mean([b9d[s][x] for x in c], "efficiency"), mean([b2d[s][x] for x in c], "efficiency")), de))
        issues.append(check(f"judge dFact {s}", delta_pp(mean([b9d[s][x] for x in c], "factulity"), mean([b2d[s][x] for x in c], "factulity")), df))
        issues.append(check(f"judge dRec {s}", delta_pp(mean([b9d[s][x] for x in c], "recall"), mean([b2d[s][x] for x in c], "recall")), dr))
        r = pearson([b2d[s][x]["recall"] for x in c], [b9d[s][x]["recall"] for x in c])
        issues.append(check(f"judge r {s}", round(r, 2), er, 0.02))

    # typical cases spot check
    issues.append(check("PMC11395317 qwen rec", b2["qwen3-8b"]["PMC11395317"]["recall"], 0.17, 0.01))
    issues.append(check("PMC11439974 ds rec", b9d["deepseek-r1"]["PMC11439974"]["recall"], 0.50, 0.01))

    # file coverage
    for judge in ["gemma-2b-it", "gemma-9b-it"]:
        for s in SUBJ:
            for g in ["direct", "inference_augmented"]:
                n = len(load_re(judge, s, g))
                if n != 100:
                    issues.append(f"COVERAGE {judge} {s} {g}: {n}/100")

    issues = [x for x in issues if x]
    if issues:
        print(f"FAILED: {len(issues)} issue(s)")
        for i in issues:
            print(f"  - {i}")
        raise SystemExit(1)
    print("OK: all checked report numbers match source data (within tolerance).")


if __name__ == "__main__":
    main()
