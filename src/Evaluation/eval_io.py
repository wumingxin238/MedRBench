"""Helpers for loading patient cases and model inference outputs."""
import json
from typing import Any, Dict


def load_model_outputs(
    filepath: str,
    model_name: str,
    embedded: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Load inference outputs for evaluation.

    Standard format (InferenceResults/oracle_diagnosis.json):
        { "PMC123": { "gemini2-ft": { "content": "...", "input": "..." } } }

    Embedded format (oracle_diagnosis_gemini.json):
        { "PMC123": { "generate_case": {...}, "gemini2-ft": { "content": "..." } } }
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if not embedded:
        return raw

    outputs: Dict[str, Dict[str, Any]] = {}
    for case_id, case in raw.items():
        if model_name in case:
            outputs[case_id] = {model_name: case[model_name]}
    return outputs
