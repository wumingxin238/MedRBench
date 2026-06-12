#!/usr/bin/env python3
"""Download Qwen3 subject models for Stage-1 (HF or ModelScope)."""

from __future__ import annotations

import argparse

MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
}


def download_hf(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id)


def download_modelscope(repo_id: str) -> str:
    from modelscope import snapshot_download

    return snapshot_download(repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["qwen3-8b", "qwen3-14b", "both"],
        default="both",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "modelscope"],
        default="hf",
        help="modelscope is often faster in CN; hf uses HF_ENDPOINT if set",
    )
    args = parser.parse_args()

    keys = list(MODELS.keys()) if args.model == "both" else [args.model]
    fn = download_hf if args.backend == "hf" else download_modelscope

    for key in keys:
        repo = MODELS[key]
        print(f"Downloading {key} -> {repo} via {args.backend} ...")
        path = fn(repo)
        print(f"  saved: {path}")


if __name__ == "__main__":
    main()
