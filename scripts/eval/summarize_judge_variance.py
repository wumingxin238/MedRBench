#!/usr/bin/env python3
import json
from pathlib import Path

docs = Path(__file__).resolve().parents[2] / "docs" / "artifacts"


def analyze(path, label):
    rows = json.load(open(path, encoding="utf-8"))
    print(f"=== {label}  n={len(rows)} ===")
    stable = sum(1 for r in rows if r["unique_hit_patterns"] == 1)
    print(f"  STABLE: {stable}/{len(rows)}")
    print(f"  max stdev: {max(r['stdev_recall'] for r in rows) * 100:.2f} pp")
    match = sum(1 for r in rows if abs(r["mean_recall"] - r["saved_recall"]) < 0.02)
    print(f"  matches saved (±2pp): {match}/{len(rows)}")
    for r in sorted(rows, key=lambda x: -abs(x["mean_recall"] - x["saved_recall"]))[:10]:
        d = (r["mean_recall"] - r["saved_recall"]) * 100
        print(
            f"    {r['id']} {r['model']:12} saved {100*r['saved_recall']:5.1f}% "
            f"now {100*r['mean_recall']:5.1f}% delta {d:+5.1f}pp"
        )


def main():
    analyze(docs / "judge_variance_A_hitonly_t0.json", "A hit_only t=0")
    print()
    analyze(docs / "judge_variance_B_full_t0.json", "B full t=0")
    print()
    analyze(docs / "judge_variance_C_hitonly_t03.json", "C hit_only t=0.3")

    A = {
        (r["id"], r["model"]): r["mean_recall"]
        for r in json.load(open(docs / "judge_variance_A_hitonly_t0.json", encoding="utf-8"))
    }
    B = {
        (r["id"], r["model"]): r["mean_recall"]
        for r in json.load(open(docs / "judge_variance_B_full_t0.json", encoding="utf-8"))
    }
    print("\n=== A hit_only vs B full (gemini, same 10 reps each) ===")
    for k in sorted(B):
        print(f"  {k[0]}: hit_only {100*A[k]:.1f}%  full {100*B[k]:.1f}%  delta {(B[k]-A[k])*100:+.1f}pp")


if __name__ == "__main__":
    main()
