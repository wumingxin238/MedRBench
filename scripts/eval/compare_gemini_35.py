#!/usr/bin/env python3
"""
Compare Gemini (gemini2-ft) on 35 test cases:
  - inference text (optional: vs paper oracle_diagnosis.json)
  - your Qwen-judge metrics (acc + reasoning)

Paper HuggingFace oracle_diagnosis.json has model OUTPUTS only, not per-case eval scores.

Usage:
  python scripts/eval/compare_gemini_35.py
  python scripts/eval/compare_gemini_35.py --paper-oracle /path/to/oracle_diagnosis.json
"""
import argparse
import json
import os
import re
from pathlib import Path


def extract_answer(text: str) -> str:
    if not text:
        return ""
    if "### Answer" in text:
        return text.split("### Answer")[-1].replace("\n", " ").replace(":", "").strip()
    return text.strip()


def count_steps(content: str) -> int:
    if not content:
        return 0
    return len(re.findall(r"<step\s+\d+>", content, re.I))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--paper-oracle", default=None, help="Downloaded oracle_diagnosis.json")
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    root = Path(args.project_root or Path(__file__).resolve().parents[2])
    test_cases = load_json(root / "data/MedRBench/test_cases.json")
    case_ids = sorted(test_cases.keys())

    local_gemini = load_json(root / "src/Inference/oracle_diagnosis_gemini.json")
    paper_oracle = None
    if args.paper_oracle and Path(args.paper_oracle).is_file():
        paper_oracle = load_json(Path(args.paper_oracle))

    acc_dir = root / "src/Evaluation/acc_results_qwen_judge/gemini2-ft"
    reason_dir = root / "src/Evaluation/reasoning_results_qwen_judge/gemini2-ft"

    rows = []
    for cid in case_ids:
        gt = test_cases[cid]["generate_case"]["diagnosis_results"]
        local_content = local_gemini.get(cid, {}).get("gemini2-ft", {}).get("content", "")
        local_ans = extract_answer(local_content)
        local_steps = count_steps(local_content)

        paper_ans = ""
        paper_steps = 0
        same_as_paper = ""
        if paper_oracle and cid in paper_oracle:
            pc = paper_oracle[cid].get("gemini2-ft", {}).get("content", "")
            paper_ans = extract_answer(pc)
            paper_steps = count_steps(pc)
            same_as_paper = "yes" if local_ans == paper_ans and local_content.strip() == pc.strip() else "no"

        acc_path = acc_dir / f"{cid}.json"
        acc_ok = None
        if acc_path.is_file():
            acc_ok = load_json(acc_path).get("accuracy")

        eff = fac = rec = None
        n_steps_eval = 0
        rpath = reason_dir / f"{cid}.json"
        if rpath.is_file():
            rd = load_json(rpath)
            eff = rd.get("efficiency")
            fac = rd.get("factulity")
            rec = rd.get("recall")
            n_steps_eval = len(rd.get("reasoning_eval") or [])

        rows.append(
            {
                "case_id": cid,
                "gt_diagnosis": gt[:80] + ("..." if len(gt) > 80 else ""),
                "gemini_answer": local_ans[:80] + ("..." if len(local_ans) > 80 else ""),
                "reasoning_steps_in_output": local_steps,
                "reasoning_steps_evaluated": n_steps_eval,
                "accuracy_qwen_judge": acc_ok,
                "efficiency": eff,
                "factuality": fac,
                "recall": rec,
                "same_inference_as_paper": same_as_paper,
                "paper_answer": (paper_ans[:60] + "...") if paper_ans else "",
            }
        )

    out_csv = Path(args.out_csv or root / "src/Evaluation/gemini2-ft_35_compare.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join('"' + str(r[h]).replace('"', '""') + '"' for h in headers) + "\n")

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    n_acc = sum(1 for r in rows if r["accuracy_qwen_judge"] is not None)
    n_reason = sum(1 for r in rows if r["efficiency"] is not None)
    print(f"Cases: {len(rows)}")
    print(f"Accuracy files: {n_acc}/35  mean accuracy: {avg('accuracy_qwen_judge'):.3f}")
    print(f"Reasoning files: {n_reason}/35")
    print(f"  mean efficiency: {avg('efficiency'):.3f}")
    print(f"  mean factuality: {avg('factuality'):.3f}")
    print(f"  mean recall:     {avg('recall'):.3f}")
    if paper_oracle:
        same = sum(1 for r in rows if r["same_inference_as_paper"] == "yes")
        print(f"Inference identical to paper gemini2-ft: {same}/{len(rows)}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
