#!/usr/bin/env python3
"""Quick coverage check: treatment_planning.json vs demo 100."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = json.loads((ROOT / "data/InferenceResults/treatment_planning.json").read_text(encoding="utf-8"))
manifest = json.loads((ROOT / "data/MedRBench/demo_stage1_manifest.json").read_text(encoding="utf-8"))
ids = manifest["treatment"]["case_ids"]

print(f"Full file cases: {len(src)}")
print(f"Demo 100 in file: {sum(1 for c in ids if c in src)}/100")
for m in ["o3-mini", "deepseek-r1", "qwen3-8b"]:
    n = sum(1 for c in ids if c in src and m in src[c])
    print(f"  {m}: {n}/100")

all_m: set[str] = set()
for c in ids:
    if c in src:
        all_m |= set(src[c].keys())
print("Models in demo subset:", sorted(all_m))
