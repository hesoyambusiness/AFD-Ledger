"""Standalone configuration helpers for fixed-catalog experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .config import load_model
from .specs import ModelSpec


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/fixed_catalog.json"


def load_experiment_config(path: Path = DEFAULT_CONFIG) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "models",
        "existing_catalog",
        "simulated_catalog",
        "bounded_search",
        "reference_cases",
        "slo_sweeps",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(
            f"fixed-catalog config is missing: {', '.join(sorted(missing))}"
        )
    return config


def load_experiment_models(
    config: Mapping[str, object],
) -> Mapping[str, ModelSpec]:
    return {
        label: load_model(
            Path(model_id),
            compute_precision=None,
            attention_precision="bf16",
            activation_precision="bf16",
            kv_precision="bf16",
        )
        for label, model_id in config["models"].items()
    }


def reference_case(
    config: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    for case in config["reference_cases"]:
        if case["name"] == name:
            return case
    raise KeyError(f"unknown reference case: {name}")
