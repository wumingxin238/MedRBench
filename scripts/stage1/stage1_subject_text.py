"""Extract reasoning / final inference from Stage-1 subject model outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

EVAL_DIR = Path(__file__).resolve().parents[2] / "src" / "Evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from utils import get_reasoning_content, split_reasoning  # noqa: E402

_ANSWER_MARKERS = (
    "### Answer:",
    "**Answer:**",
    "### Final diagnosis:",
    "**Final diagnosis:**",
)


def _strip_fences(text: str) -> str:
    return text.replace("```", "").strip()


def extract_final_answer(result: dict) -> str:
    """Parse subject model final diagnosis / treatment answer from content."""
    content = _strip_fences(result.get("content") or "")
    for marker in _ANSWER_MARKERS:
        if marker in content:
            return content.split(marker, 1)[1].strip()
    return ""


def get_content_for_split(result: dict, subject_model: str) -> str:
    """Pick the text field used for <step N> splitting (MedRBench convention)."""
    if subject_model == "deepseek-r1-thinkingprocess" and result.get("thinking_process"):
        return result["thinking_process"]
    if result.get("thinking_process") and not result.get("content"):
        return result["thinking_process"]
    return result.get("content") or ""


def build_reasoning_steps(
    result: dict,
    subject_model: str,
    *,
    group: str,
    max_steps: int = 10,
) -> Tuple[List[str], str]:
    """
    Build step list and combined string for reasoning evaluation.

    group:
      - direct: reasoning steps only
      - inference_augmented: reasoning steps + final model inference as last step
    """
    source = get_content_for_split(result, subject_model)
    steps = split_reasoning(source, max_steps=max_steps)
    answer = extract_final_answer(result)

    if group == "inference_augmented" and answer:
        steps = steps + [f"Final model inference: {answer}"]

    combined = "\n".join(steps)
    return steps, combined


def gt_reasoning_for_case(case: dict, task: str) -> str:
    """Ground-truth reasoning text for completeness (diagnosis vs treatment)."""
    gen = case["generate_case"]
    if task == "treatment":
        return gen.get("treatment_plan", "") or gen.get("final_treatment", "")
    diff = gen.get("differential_diagnosis", "")
    final = gen.get("final_diagnosis", "")
    return f"{diff}\n Final diagnosis:\n{final}"


def gt_answer_for_case(case: dict, task: str) -> str:
    gen = case["generate_case"]
    if task == "treatment":
        return gen.get("final_treatment", "") or gen.get("treatment_plan", "")
    return gen.get("final_diagnosis", "")
