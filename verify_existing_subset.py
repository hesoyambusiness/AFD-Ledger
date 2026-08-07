#!/usr/bin/env python3
"""Generate the outcome-conditioned existing-hardware availability subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.catalog_provisioning import provision_catalog
from sim.catalog_subset import select_existing_subset
from sim.experiment_config import (
    load_experiment_config,
    load_experiment_models,
    reference_case,
)
from sim.provisioning import DeploymentSpecification
from sim.throughput_catalog import load_priced_existing_catalog


def verify(output_dir: Path) -> dict:
    config = load_experiment_config()
    models = load_experiment_models(config)
    subset_config = config["existing_catalog"]
    names = tuple(subset_config["prices_usd_per_hour"])
    catalog = dict(load_priced_existing_catalog(config, names))
    case = reference_case(config, "existing_deepseek")
    specification = DeploymentSpecification(
        model=models["DeepSeek-V3.2"],
        seq_len=int(case["seq_len"]),
        tpot_slo_ms=float(case["tpot_slo_ms"]),
        budget=float(case["budget"]),
        hardware_menu=catalog,
    )
    full = provision_catalog(specification)
    subset = select_existing_subset(
        full,
        names,
        min_size=2,
        max_size=len(names),
        minimum_gain=float(subset_config["subset_minimum_global_gain"]),
    )
    row = {
        "model": "DeepSeek-V3.2",
        "seq_len": specification.seq_len,
        "tpot_slo_ms": specification.tpot_slo_ms,
        "budget": specification.budget,
        "hardware_subset": list(subset.hardware),
        "best_afd_attention_hardware": subset.best_afd.attention_hardware,
        "best_afd_expert_hardware": subset.best_afd.expert_hardware,
        "best_afd_throughput_tps": subset.best_afd.throughput_tps,
        "best_collocated_hardware": subset.best_collocated.attention_hardware,
        "best_collocated_throughput_tps": subset.best_collocated.throughput_tps,
        "afd_over_collocated": subset.comparison_gain,
        "selection_scope": (
            "existing hardware only; maximum-cardinality availability subset "
            "meeting the configured minimum gain"
        ),
        "full_catalog_size": len(names),
        "selected_subset_size": len(subset.hardware),
        "minimum_global_gain": subset_config["subset_minimum_global_gain"],
        "price_snapshot_date": subset_config["price_snapshot_date"],
        "price_source": subset_config["price_source"],
        "excluded_hardware": [name for name in names if name not in subset.hardware],
        "interpretation": (
            "valid only when excluded devices are unavailable; not a reason to "
            "ignore stronger collocated hardware that is actually procurable"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "existing_hardware_subset.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(row, handle, indent=2)
        handle.write("\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/existing_hardware_subset"),
    )
    args = parser.parse_args()
    row = verify(args.output_dir)
    print(
        f"subset={'+'.join(row['hardware_subset'])} "
        f"afd={row['best_afd_attention_hardware']}+"
        f"{row['best_afd_expert_hardware']} "
        f"collocated={row['best_collocated_hardware']} "
        f"G={row['afd_over_collocated']:.6f}"
    )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
