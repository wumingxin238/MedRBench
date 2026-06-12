#!/usr/bin/env python3
"""Re-parse Gemma Scope JSON outputs with improved parser (no GPU)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional


def parse_score_json(text: str) -> Optional[dict[str, Any]]:
    text = text.replace("```json", "").replace("```", "").strip()
    if "<integer" in text.lower() or "<one short" in text.lower():
        text = re.sub(r"<[^>]+>", "", text)
    for match in re.finditer(r"\{[^{}]*\"score\"[^{}]*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            if obj.get("score") is not None:
                score = int(obj["score"])
                if 1 <= score <= 5:
                    return {"score": score, "rationale": str(obj.get("rationale", ""))}
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    for pat in (
        r'"score"\s*:\s*([1-5])',
        r"score\s*[=:]\s*([1-5])\b",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            score = int(m.group(1))
            rat = re.search(r'"rationale"\s*:\s*"([^"]*)"', text)
            return {"score": score, "rationale": rat.group(1) if rat else ""}
    return None


def reparse_file(path: Path, in_place: bool) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    fixed = 0
    for case in data.get("cases", {}).values():
        for group, score in case.get("scores", {}).items():
            if score.get("parsed"):
                continue
            raw = score.get("raw_response", "")
            parsed = parse_score_json(raw)
            if parsed:
                score["parsed"] = parsed
                score["reparsed"] = True
                fixed += 1
    out = path if in_place else path.with_name(path.stem + "_reparsed.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{path.name}: recovered {fixed} scores -> {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    for p in args.paths:
        reparse_file(p, args.in_place)


if __name__ == "__main__":
    main()
