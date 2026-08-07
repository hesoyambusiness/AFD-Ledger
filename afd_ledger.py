#!/usr/bin/env python3
"""Provision a fixed hardware catalog for collocated or AFD serving."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

from sim.catalog_provisioning import provision_catalog
from sim.config import load_model
from sim.experiment_config import (
    DEFAULT_CONFIG,
    load_experiment_config,
    load_experiment_models,
)
from sim.guided_catalog_search import guided_provision_catalog
from sim.provisioning import DeploymentSpecification, HardwareOption
from sim.simulated_catalog import fixed_simulated_catalog
from sim.throughput_catalog import load_priced_existing_catalog


def build_catalog(
    configuration: Mapping[str, object],
    catalog_name: str,
) -> Mapping[str, HardwareOption]:
    commercial = load_priced_existing_catalog(configuration)
    if catalog_name == "existing":
        return commercial
    if catalog_name == "constructed-existing":
        names = configuration["existing_catalog"]["constructed_availability_subset"]
        return {name: commercial[name] for name in names}
    if catalog_name == "simulated":
        return fixed_simulated_catalog(
            commercial,
            configuration["simulated_catalog"],
        )
    raise ValueError(f"unknown catalog: {catalog_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--catalog",
        choices=("existing", "constructed-existing", "simulated"),
        default="existing",
        help=(
            "fixed catalog to provision; 'constructed-existing' is the "
            "outcome-conditioned paper availability scenario"
        ),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        help="optional fixed subset of device names from the selected catalog",
    )
    parser.add_argument(
        "--model",
        default="DeepSeek-V3.2",
        help="model label from the experiment config, or a local model config/path",
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--tpot-slo-ms", type=float, default=12.0)
    parser.add_argument("--budget", type=float, default=48.0)
    parser.add_argument("--evaluation-budget", type=int, default=20)
    parser.add_argument("--retained-width", type=int, default=2)
    parser.add_argument("--expansion-width", type=int, default=6)
    parser.add_argument("--neighbor-round-limit", type=int, default=3)
    parser.add_argument(
        "--exact",
        action="store_true",
        help="enumerate every role-compatible ordered AFD assignment",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configuration = load_experiment_config(args.config)
    models = load_experiment_models(configuration)
    catalog = build_catalog(configuration, args.catalog)
    if args.devices:
        requested = tuple(dict.fromkeys(args.devices))
        missing = [name for name in requested if name not in catalog]
        if missing:
            parser.error(
                "unknown device(s) for the selected catalog: "
                + ", ".join(missing)
            )
        catalog = {name: catalog[name] for name in requested}
    if args.model in models:
        model = models[args.model]
    else:
        model = load_model(
            Path(args.model),
            compute_precision=None,
            attention_precision="bf16",
            activation_precision="bf16",
            kv_precision="bf16",
        )
    specification = DeploymentSpecification(
        model=model,
        seq_len=args.seq_len,
        tpot_slo_ms=args.tpot_slo_ms,
        budget=args.budget,
        hardware_menu=catalog,
    )
    contract = {
        "model": specification.model.name,
        "workload": specification.workload,
        "seq_len": specification.seq_len,
        "tpot_slo_ms": specification.tpot_slo_ms,
        "budget": specification.budget,
        "hardware_catalog": list(specification.hardware_menu),
        "constraints": asdict(specification.constraints),
    }

    if args.exact:
        try:
            result = provision_catalog(specification)
        except ValueError as error:
            parser.error(str(error))
        payload = {
            "search": "exact",
            "specification": contract,
            "selected_architecture": result.selected_architecture,
            "global_gain": result.comparison_gain,
            "selected_plan": asdict(result.selected_plan),
            "best_afd": asdict(result.best_afd) if result.best_afd else None,
            "best_collocated": asdict(result.best_collocated),
            "feasible_afd_assignments": len(result.afd_plans),
        }
    else:
        try:
            result = guided_provision_catalog(
                specification,
                evaluation_budget=args.evaluation_budget,
                beam_width=args.retained_width,
                expansion_width=args.expansion_width,
                max_rounds=args.neighbor_round_limit,
            )
        except ValueError as error:
            parser.error(str(error))
        payload = {
            "search": "bounded",
            "specification": contract,
            "selected_architecture": result.selected_architecture,
            "global_gain": result.comparison_gain,
            "selected_plan": asdict(result.selected_plan),
            "best_afd": asdict(result.best_afd) if result.best_afd else None,
            "best_collocated": asdict(result.best_collocated),
            "valid_ordered_assignments": result.valid_assignments,
            "evaluation_budget": result.evaluation_budget,
            "attempted_assignments": [
                {
                    "attention_hardware": attention,
                    "expert_hardware": expert,
                    "feasible": (attention, expert) in result.evaluated_plans,
                }
                for attention, expert in result.evaluation_order
            ],
        }

    print(
        f"selected={payload['selected_architecture']} "
        f"throughput={payload['selected_plan']['throughput_tps']:.3f} "
        f"tpot={payload['selected_plan']['tpot_ms']:.3f}ms "
        f"global_gain={payload['global_gain']:.6f}"
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
