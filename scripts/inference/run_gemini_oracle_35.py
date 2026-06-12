#!/usr/bin/env python3
"""
Re-run oracle diagnosis inference for 35 test cases with gemini-2.5-flash-thinking.

Reads:  data/MedRBench/test_cases.json
Writes: src/Inference/oracle_diagnosis_gemini.json (embedded format)

API (env, optional):
  GEMINI_URL or OPENAI_COMPAT_BASE_URL  (default: https://xiaoai.plus/v1)
  GEMINI_API_KEY or OPENAI_COMPAT_API_KEY

Usage:
  python scripts/inference/run_gemini_oracle_35.py
  python scripts/inference/run_gemini_oracle_35.py --force          # overwrite all 35
  python scripts/inference/run_gemini_oracle_35.py --workers 4
  python scripts/inference/run_gemini_oracle_35.py --case PMC11625232
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

MODEL_ID = "gemini-2.5-flash-thinking"
MODEL_KEY = "gemini2-ft"
SYSTEM_PROMPT = "You are a professional doctor"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def make_client() -> OpenAI:
    base_url = os.environ.get("GEMINI_URL") or os.environ.get(
        "OPENAI_COMPAT_BASE_URL", "https://xiaoai.plus/v1"
    )
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_COMPAT_API_KEY")
    if not api_key:
        # Fallback: same as oracle_diagnose.py when env not set
        infer_dir = project_root() / "src" / "Inference"
        sys.path.insert(0, str(infer_dir))
        import oracle_diagnose as od  # noqa: WPS433

        api_key = od.GEMINI_API_KEY
        base_url = od.GEMINI_URL
    if not api_key:
        raise SystemExit(
            "Set GEMINI_API_KEY or OPENAI_COMPAT_API_KEY (or configure src/Inference/oracle_diagnose.py)."
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def query_gemini(client: OpenAI, prompt: str, max_retries: int = 5) -> str | None:
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            attempt += 1
            if "429" in err or "500" in err or "502" in err or "503" in err:
                wait = min(30 * attempt, 120)
                print(f"Retry {attempt}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
                continue
            print(f"API error (no retry): {e}")
            return None
    print(f"Failed after {max_retries} retries")
    return None


def needs_run(record: dict, force: bool) -> bool:
    if force:
        return True
    block = record.get(MODEL_KEY) or {}
    content = block.get("content")
    return not content


def process_one(client: OpenAI, case_id: str, case: dict, prompt_template: str):
    patient_case = case["generate_case"]["case_summary"]
    prompt = prompt_template.format(case=patient_case)
    content = query_gemini(client, prompt)
    return case_id, {"input": prompt, "content": content}


def main():
    parser = argparse.ArgumentParser(description="Gemini oracle inference on 35 test cases")
    parser.add_argument("--force", action="store_true", help="Re-run all cases even if content exists")
    parser.add_argument("--workers", type=int, default=1, help="Parallel API calls (default 1, safer for proxies)")
    parser.add_argument("--case", action="append", dest="cases", help="Only run these PMC IDs")
    args = parser.parse_args()

    root = project_root()
    test_path = root / "data" / "MedRBench" / "test_cases.json"
    out_path = root / "src" / "Inference" / "oracle_diagnosis_gemini.json"
    template_path = root / "src" / "Inference" / "instructions" / "oracle_diagnose.txt"

    test_cases = load_json(test_path)
    prompt_template = template_path.read_text(encoding="utf-8")

    if out_path.is_file():
        merged = load_json(out_path)
        for cid, case in test_cases.items():
            if cid not in merged:
                merged[cid] = case
            else:
                for k, v in case.items():
                    if k != MODEL_KEY:
                        merged[cid][k] = v
    else:
        merged = {cid: dict(case) for cid, case in test_cases.items()}

    if args.cases:
        todo = [c for c in args.cases if c in test_cases]
        missing = set(args.cases) - set(todo)
        if missing:
            print("Unknown case IDs:", ", ".join(sorted(missing)))
    else:
        todo = sorted(test_cases.keys())

    run_ids = [cid for cid in todo if needs_run(merged.get(cid, test_cases[cid]), args.force)]
    skip = len(todo) - len(run_ids)
    if skip:
        print(f"Skipping {skip} case(s) with existing content (use --force to re-run).")
    if not run_ids:
        print("Nothing to run.")
        return

    print(f"Model: {MODEL_ID}")
    print(f"Cases to run: {len(run_ids)}/{len(todo)}")
    print(f"Output: {out_path}")

    client = make_client()
    workers = max(1, args.workers)
    save_lock = threading.Lock()

    def run_and_save(case_id: str):
        _, result = process_one(client, case_id, test_cases[case_id], prompt_template)
        if not result.get("content"):
            return case_id, False
        with save_lock:
            merged[case_id] = {**test_cases[case_id], MODEL_KEY: result}
            save_json(out_path, merged)
        return case_id, True

    ok = fail = 0
    if workers == 1:
        for cid in tqdm(run_ids, desc="Gemini oracle"):
            _, success = run_and_save(cid)
            ok += int(success)
            fail += int(not success)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_and_save, cid): cid for cid in run_ids}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Gemini oracle"):
                _, success = fut.result()
                ok += int(success)
                fail += int(not success)

    print(f"Done. success={ok} failed={fail} -> {out_path}")


if __name__ == "__main__":
    main()
