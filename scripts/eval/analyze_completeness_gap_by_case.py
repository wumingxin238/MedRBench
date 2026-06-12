#!/usr/bin/env python3
"""Per-case completeness (recall) gap: qwq vs gemini2-ft under same Qwen judge."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "src/Evaluation"
GEMINI_DIR = EV / "reasoning_results_qwen_judge_paper_957/gemini2-ft"
QWQ_DIR = EV / "reasoning_results_qwen_judge_paper_957/qwq"
OUT_CSV = ROOT / "docs/artifacts/completeness_gap_by_case.csv"
OUT_MD = ROOT / "docs/reports/COMPLETENESS_GAP_CASE_ANALYSIS.md"


def load_recall(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    gt = d.get("gt_reasoning_eval") or []
    return {
        "id": d.get("id", path.stem),
        "recall": float(d.get("recall", 0) or 0),
        "efficiency": float(d.get("efficiency", 0) or 0),
        "factuality": float(d.get("factulity", 0) or 0),
        "n_gt": len(gt),
        "n_hit": sum(1 for x in gt if x.get("hit")),
        "gt_steps": [x.get("step", "") for x in gt],
        "hits": [bool(x.get("hit")) for x in gt],
        "n_pred_steps": len(d.get("reasoning_eval") or []),
    }


def gt_step_text_match(a: list[str], b: list[str]) -> bool:
    if len(a) != len(b):
        return False
    return all(x.strip() == y.strip() for x, y in zip(a, b))


def leave_top_k_out_mean(deltas: list[float], k: int) -> float:
    if k <= 0:
        return statistics.mean(deltas)
    sorted_abs = sorted(range(len(deltas)), key=lambda i: abs(deltas[i]), reverse=True)
    keep = [deltas[i] for i in range(len(deltas)) if i not in sorted_abs[:k]]
    return statistics.mean(keep) if keep else float("nan")


def concentration_top_pct(deltas: list[float], pct: float) -> float:
    """Share of total positive delta (qwq-gemini) from top pct fraction by delta."""
    pos = [(i, d) for i, d in enumerate(deltas) if d > 0]
    if not pos:
        return 0.0
    pos.sort(key=lambda x: x[1], reverse=True)
    total = sum(d for _, d in pos)
    n_top = max(1, int(len(pos) * pct / 100))
    return sum(d for _, d in pos[:n_top]) / total


def main() -> None:
    gemini_files = {p.stem: p for p in GEMINI_DIR.glob("PMC*.json")}
    qwq_files = {p.stem: p for p in QWQ_DIR.glob("PMC*.json")}
    common = sorted(set(gemini_files) & set(qwq_files))
    if not common:
        raise SystemExit("No paired PMC files found.")

    rows = []
    for cid in common:
        g = load_recall(gemini_files[cid])
        q = load_recall(qwq_files[cid])
        delta = q["recall"] - g["recall"]
        same_gt = gt_step_text_match(g["gt_steps"], q["gt_steps"])
        # Step-level hit agreement (when GT text identical)
        hit_agree = hit_disagree = 0
        if same_gt and g["gt_steps"]:
            for hg, hq in zip(g["hits"], q["hits"]):
                if hg == hq:
                    hit_agree += 1
                else:
                    hit_disagree += 1
        rows.append(
            {
                "id": cid,
                "gemini_recall": g["recall"],
                "qwq_recall": q["recall"],
                "delta": delta,
                "gemini_hit": g["n_hit"],
                "qwq_hit": q["n_hit"],
                "n_gt": g["n_gt"],
        "qwq_n_gt": q["n_gt"],
                "same_gt_steps": same_gt,
                "hit_step_agree": hit_agree,
                "hit_step_disagree": hit_disagree,
                "gemini_pred_steps": g["n_pred_steps"],
                "qwq_pred_steps": q["n_pred_steps"],
                "gemini_eff": g["efficiency"],
                "qwq_eff": q["efficiency"],
            }
        )

    deltas = [r["delta"] for r in rows]
    gemini_recalls = [r["gemini_recall"] for r in rows]
    qwq_recalls = [r["qwq_recall"] for r in rows]

    mean_g = statistics.mean(gemini_recalls)
    mean_q = statistics.mean(qwq_recalls)
    mean_delta = mean_q - mean_g

    # Pattern buckets
    buckets = {
        "qwq_high_gemini_low (q>=0.7, g<0.3)": 0,
        "qwq_low_gemini_high (q<0.3, g>=0.7)": 0,
        "both_zero": 0,
        "both_high (>=0.7)": 0,
        "moderate_gap (|delta|>=0.5)": 0,
        "qwq_better (delta>0)": 0,
        "gemini_better (delta<0)": 0,
        "tie (|delta|<0.01)": 0,
    }
    for r in rows:
        g, q, d = r["gemini_recall"], r["qwq_recall"], r["delta"]
        if g < 0.01 and q < 0.01:
            buckets["both_zero"] += 1
        if q >= 0.7 and g < 0.3:
            buckets["qwq_high_gemini_low (q>=0.7, g<0.3)"] += 1
        if q < 0.3 and g >= 0.7:
            buckets["qwq_low_gemini_high (q<0.3, g>=0.7)"] += 1
        if g >= 0.7 and q >= 0.7:
            buckets["both_high (>=0.7)"] += 1
        if abs(d) >= 0.5:
            buckets["moderate_gap (|delta|>=0.5)"] += 1
        if d > 0.01:
            buckets["qwq_better (delta>0)"] += 1
        elif d < -0.01:
            buckets["gemini_better (delta<0)"] += 1
        else:
            buckets["tie (|delta|<0.01)"] += 1

    same_gt_count = sum(1 for r in rows if r["same_gt_steps"])
    total_step_disagree = sum(r["hit_step_disagree"] for r in rows if r["same_gt_steps"])

    # Correlation
    n = len(rows)
    mg, mq = mean_g, mean_q
    cov = sum((gemini_recalls[i] - mg) * (qwq_recalls[i] - mq) for i in range(n)) / n
    sg = (sum((x - mg) ** 2 for x in gemini_recalls) / n) ** 0.5
    sq = (sum((x - mq) ** 2 for x in qwq_recalls) / n) ** 0.5
    pearson = cov / (sg * sq) if sg * sq else 0.0
    gemini_zero = sum(1 for x in gemini_recalls if x < 0.01)
    qwq_zero = sum(1 for x in qwq_recalls if x < 0.01)
    g0_q70 = sum(
        1 for r in rows if r["gemini_recall"] < 0.01 and r["qwq_recall"] >= 0.7
    )

    top_qwq_wins = sorted(rows, key=lambda r: r["delta"], reverse=True)[:20]
    top_gemini_wins = sorted(rows, key=lambda r: r["delta"])[:20]

    # Outlier contribution
    loo = {k: leave_top_k_out_mean(deltas, k) for k in (0, 5, 10, 20, 50, 100)}
    conc = {p: concentration_top_pct(deltas, p) for p in (1, 5, 10, 20)}

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Completeness 逐例差距分析（qwq vs gemini2-ft · Qwen Judge）",
        "",
        f"配对病例：**{len(rows)}** · 生成脚本：`scripts/eval/analyze_completeness_gap_by_case.py`",
        "",
        "## 1. 聚合 vs 逐例",
        "",
        "| 指标 | gemini2-ft | qwq | Δ (qwq−gemini) |",
        "|------|------------|-----|----------------|",
        f"| 均值 Completeness | {100*mean_g:.2f}% | {100*mean_q:.2f}% | {100*mean_delta:+.2f} pp |",
        f"| 中位 Completeness | {100*statistics.median(gemini_recalls):.2f}% | {100*statistics.median(qwq_recalls):.2f}% | — |",
        f"| 标准差 | {100*statistics.stdev(gemini_recalls):.2f} pp | {100*statistics.stdev(qwq_recalls):.2f} pp | {100*statistics.stdev(deltas):.2f} pp (Δ) |",
        f"| 逐例 Pearson r | — | — | **{pearson:.3f}**（弱相关） |",
        "",
        f"- gemini recall=0：**{gemini_zero}** 例 ({100*gemini_zero/len(rows):.1f}%)",
        f"- qwq recall=0：**{qwq_zero}** 例 ({100*qwq_zero/len(rows):.1f}%)",
        f"- gemini=0 且 qwq≥70%：**{g0_q70}** 例 ({100*g0_q70/len(rows):.1f}%)",
        "",
        "**结论预览**：若差距由少数 outlier 造成，去掉 top-k 最大正 Δ 后均值应明显回落。",
        "",
        "## 2. 去掉最大 outlier 后的均值 Δ",
        "",
        "| 去掉 |Δ| 最大的 k 例后，剩余均值 Δ (qwq−gemini) |",
        "|-----|----------------------------------------|",
    ]
    for k, v in loo.items():
        lines.append(f"| k={k} | {100*v:+.2f} pp |")

    lines.extend(
        [
            "",
            f"去掉 top-50 后均值 Δ 仍为 **{100*loo[50]:+.2f} pp**（全量 {100*mean_delta:+.2f} pp）→ "
            + (
                "**整体分布偏移**，不是极少数病例单独造成。"
                if abs(loo[50]) > 0.5 * abs(mean_delta)
                else "**部分由 outlier 拉动**，去掉后差距明显缩小。"
            ),
            "",
            "## 3. 正 Δ 集中度（qwq 更高的病例）",
            "",
            "| top 病例占比 | 贡献占全部正 Δ 的比例 |",
            "|-------------|------------------------|",
        ]
    )
    for p, c in conc.items():
        lines.append(f"| top {p}% | {100*c:.1f}% |")

    lines.extend(["", "## 4. 逐例模式分布", "", "| 模式 | 病例数 | 占比 |", "|------|--------|------|"])
    for name, cnt in buckets.items():
        lines.append(f"| {name} | {cnt} | {100*cnt/len(rows):.1f}% |")

    lines.extend(
        [
            "",
            "## 5. GT 步骤一致性",
            "",
            f"- 两模型结果中 GT 步骤文本完全一致：**{same_gt_count}/{len(rows)}** ({100*same_gt_count/len(rows):.1f}%)",
            f"- 在 GT 一致的病例上，逐步 hit 判定不一致总数：**{total_step_disagree}** 步",
            "- 同一 GT 步、不同模型推理文本 → hit 不同，说明差距主要来自**预测链表述**而非 GT 拆分随机性。",
            "",
            "## 6. Top 20：qwq 明显高于 gemini (Δ recall)",
            "",
            "| PMC | gemini | qwq | Δ | gemini hits | qwq hits | GT steps |",
            "|-----|--------|-----|---|-------------|----------|----------|",
        ]
    )
    for r in top_qwq_wins:
        lines.append(
            f"| {r['id']} | {100*r['gemini_recall']:.0f}% | {100*r['qwq_recall']:.0f}% | "
            f"{100*r['delta']:+.0f}pp | {r['gemini_hit']}/{r['n_gt']} | {r['qwq_hit']}/{r['qwq_n_gt']} | {r['n_gt']}/{r['qwq_n_gt']} |"
        )

    lines.extend(
        [
            "",
            "## 7. Top 20：gemini 明显高于 qwq",
            "",
            "| PMC | gemini | qwq | Δ | gemini hits | qwq hits |",
            "|-----|--------|-----|---|-------------|----------|",
        ]
    )
    for r in top_gemini_wins:
        lines.append(
            f"| {r['id']} | {100*r['gemini_recall']:.0f}% | {100*r['qwq_recall']:.0f}% | "
            f"{100*r['delta']:+.0f}pp | {r['gemini_hit']}/{r['n_gt']} | {r['qwq_hit']}/{r['qwq_n_gt']} |"
        )

    same_gt_rows = [r for r in rows if r["same_gt_steps"]]
    diff_gt_rows = [r for r in rows if not r["same_gt_steps"]]
    mean_delta_same_gt = statistics.mean([r["delta"] for r in same_gt_rows]) if same_gt_rows else float("nan")
    mean_delta_diff_gt = statistics.mean([r["delta"] for r in diff_gt_rows]) if diff_gt_rows else float("nan")

    def _means(subset):
        if not subset:
            return 0, 0, 0
        return (
            statistics.mean([r["gemini_recall"] for r in subset]),
            statistics.mean([r["qwq_recall"] for r in subset]),
            statistics.mean([r["delta"] for r in subset]),
        )

    sg_g, sg_q, sg_d = _means(same_gt_rows)
    dg_g, dg_q, dg_d = _means(diff_gt_rows)

    lines.extend(
        [
            "",
            "## 8. GT 拆分一致 vs 不一致（**关键 confound**）",
            "",
            "gemini 与 qwq 为**两次独立评估**，`split_ground_truth_reasoning` 每例各跑一次；",
            f"仅 **{len(same_gt_rows)}/{len(rows)} ({100*len(same_gt_rows)/len(rows):.1f}%)** 两文件的 GT 步骤文本完全一致。",
            "",
            "| 子集 | n | mean gemini | mean qwq | mean Δ |",
            "|------|---|-------------|----------|--------|",
            f"| GT 文本一致 | {len(same_gt_rows)} | {100*sg_g:.1f}% | {100*sg_q:.1f}% | **{100*sg_d:+.1f} pp** |",
            f"| GT 文本不一致 | {len(diff_gt_rows)} | {100*dg_g:.1f}% | {100*dg_q:.1f}% | **{100*dg_d:+.1f} pp** |",
            f"| 全量 | {len(rows)} | {100*mean_g:.1f}% | {100*mean_q:.1f}% | {100*mean_delta:+.1f} pp |",
            "",
            "**解读**：",
            "- 在 **GT 完全一致** 的 84 例上，qwq 仅比 gemini 高约 **+5 pp**（两者 recall 仍都偏低 ~30%）。",
            "- 全量 +42 pp 差距中，**~37 pp 来自两次评估 GT 拆分不同**（分母 M / 步骤措辞变化），",
            "  而非单纯「qwq 推理文本更容易 hit」。",
            "- 跨模型 Completeness 对比应 **固定同一份 GT 步骤**（见 §9 方差脚本 `--mode hit_only`）。",
            "",
            "## 9. Judge 随机性 / 方差探测（设计说明）",
            "",
            "评估默认 `EVAL_TEMPERATURE=0`（greedy decoding），**理论上同一输入应得到相同 hit 判定**。",
            "若重复跑仍波动，可能来自 GPU 非确定性或 GT split 阶段（`split_ground_truth_reasoning`）的输出变化。",
            "",
            "**注意**：gemini 与 qwq 的评估是**两次独立跑**，GT 步骤文本仅 **84/957 (8.8%)** 完全一致；",
            "其余病例的分母 M 可能不同，逐例 Δ 会混入「GT 拆分差异」。",
            "",
            "在 P100 服务器上（Judge 已启动）可复跑 hit-check：",
            "",
            "```bash",
            "source scripts/server/config/eval_config.env",
            "# temperature=0：检验是否完全稳定",
            "python scripts/eval/judge_completeness_variance.py \\",
            "  --from-analysis gemini_zero --n 5 --repeats 10 --mode hit_only",
            "",
            "# temperature=0.3 + seed：检验采样方差",
            "EVAL_TEMPERATURE=0.3 python scripts/eval/judge_completeness_variance.py \\",
            "  --cases PMC11321471 PMC11625232 --repeats 10 --temperature 0.3 --seed 42",
            "```",
            "",
            "输出：`docs/artifacts/judge_completeness_variance.json`",
            "",
            "完整 CSV：`docs/artifacts/completeness_gap_by_case.csv`",
        ]
    )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Console summary
    print("=" * 72)
    print(f"Paired cases: {len(rows)}")
    print(f"Mean recall  gemini {100*mean_g:.2f}%  qwq {100*mean_q:.2f}%  delta {100*mean_delta:+.2f} pp")
    print(f"Median delta {100*statistics.median(deltas):+.2f} pp  stdev {100*statistics.stdev(deltas):.2f} pp")
    print("\nLeave-top-k-out mean delta (pp):")
    for k, v in loo.items():
        print(f"  k={k:3d} -> {100*v:+.2f}")
    print("\nBuckets:")
    for name, cnt in buckets.items():
        print(f"  {name}: {cnt} ({100*cnt/len(rows):.1f}%)")
    print(f"\nGT same: n={len(same_gt_rows)} delta={100*sg_d:+.2f}pp  |  GT diff: n={len(diff_gt_rows)} delta={100*dg_d:+.2f}pp")
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
