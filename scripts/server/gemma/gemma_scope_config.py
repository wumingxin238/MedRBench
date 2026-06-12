"""Gemma Scope defaults (aligned with demo_stage1_manifest.json)."""

GEMMA_2B = {
    "model_id": "google/gemma-2-2b",
    "release": "gemma-scope-2b-pt-res-canonical",
    "sae_id": "layer_12/width_16k/canonical",
    "sae_layer": 12,
}

GEMMA_9B = {
    "model_id": "google/gemma-2-9b",
    "release": "gemma-scope-9b-pt-res-canonical",
    "sae_id": "layer_12/width_16k/canonical",
    "sae_layer": 12,
}

DEFAULT_TOP_K = 10
DEFAULT_MAX_NEW_TOKENS = 256
