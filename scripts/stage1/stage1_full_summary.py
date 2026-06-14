#!/usr/bin/env python3
"""Stage-1 full summary: reasoning_eval + coverage check."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RE_DIR = PROJECT_ROOT / "data" / "Stage1" / "reasoning_eval"
SUBJECTS = ["o3-mini", "deepseek-r1", "qwen3-8b"]
METRICS = ["efficiency", "factulity", "recall"]


def load_ok(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [c for c in doc["cases"].values() if c.get("status") == "ok"]


def score_stats(rows: list[dict], label: str) -> None:
    print(f"\n  [{label}] n={len(rows)}")
    for k, lab in zip(METRICS, ["Eff", "Fact", "Rec"]):
        v = [r[k] for r in rows]
        perfect = sum(1 for x in v if x >= 0.999)
        print(
            f"    {lab}: mean={st.mean(v):.3f} min={min(v):.3f} max={max(v):.3f}  "
            f">=0.999: {perfect}/{len(v)} ({100*perfect/len(v):.0f}%)"
        )
    triple = sum(1 for r in rows if all(r[k] >= 0.999 for k in METRICS))
    print(f"    triple>=0.999: {triple}/{len(rows)} ({100*triple/len(rows):.0f}%)")


def main() -> None:
    print("=" * 70)
    print("STAGE-1 reasoning_eval coverage")
    print("=" * 70)
    for p in sorted(RE_DIR.glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        ok = sum(1 for c in doc["cases"].values() if c.get("status") == "ok")
        err = sum(1 for c in doc["cases"].values() if c.get("status") == "error")
        m = doc.get("meta", {})
        print(
            f"  {p.name}\n"
            f"    {ok}/100 ok  eff={m.get('mean_efficiency')} fact={m.get('mean_factulity')} rec={m.get('mean_recall')}"
        )

    print("\n" + "=" * 70)
    print("Gemma-9B qwen3-8b score distribution (completed 100/100)")
    print("=" * 70)
    for g in ["direct", "inference_augmented"]:
        score_stats(
            load_ok(RE_DIR / f"diagnosis_gemma-9b-it_qwen3-8b_{g}.json"),
            g,
        )

    print("\n" + "=" * 70)
    print("Gemma-2B: group effect (direct -> aug, mean delta pp)")
    print("=" * 70)
    for s in SUBJECTS:
        d0 = load_ok(RE_DIR / f"diagnosis_gemma-2b-it_{s}_direct.json")
        d1 = load_ok(RE_DIR / f"diagnosis_gemma-2b-it_{s}_inference_augmented.json")
        print(f"  {s}:")
        for k, lab in zip(METRICS, ["eff", "fact", "rec"]):
            m0 = st.mean(r[k] for r in d0)
            m1 = st.mean(r[k] for r in d1)
            print(f"    {lab}: {m0:.3f} -> {m1:.3f}  ({(m1-m0)*100:+.1f} pp)")


if __name__ == "__main__":
    main()
