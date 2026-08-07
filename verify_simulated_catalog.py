#!/usr/bin/env python3
"""Verify the fixed role-specialized catalog and its reference deployments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sim.catalog_provisioning import provision_catalog
from sim.experiment_config import load_experiment_config, load_experiment_models
from sim.provisioning import DeploymentSpecification
from sim.simulated_catalog import fixed_simulated_catalog
from sim.throughput_catalog import load_priced_existing_catalog


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify(output_dir: Path) -> list[dict]:
    config = load_experiment_config()
    models = load_experiment_models(config)
    commercial = load_priced_existing_catalog(config)
    catalog = dict(
        fixed_simulated_catalog(commercial, config["simulated_catalog"])
    )
    if len(catalog) != 12:
        raise AssertionError(f"expected 12 fixed catalog entries, found {len(catalog)}")
    if sum(option.supports_collocated for option in catalog.values()) != 10:
        raise AssertionError("expected ten collocated-compatible general devices")

    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_rows = []
    for name, option in sorted(catalog.items()):
        hardware = option.hardware
        catalog_rows.append(
            {
                "name": name,
                "price_h200_normalized": option.price,
                "tags": sorted(option.tags),
                "supports_attention": option.supports_attention,
                "supports_expert": option.supports_expert,
                "supports_collocated": option.supports_collocated,
                "memory_bytes": hardware.hbm_capacity,
                "hbm_bytes_per_s": hardware.hbm_bw,
                "peak_flops": dict(hardware.peak_flops),
                "intra_node_bytes_per_s": hardware.intra_node_bw,
                "inter_node_bytes_per_s": hardware.inter_node_bw,
                "inter_rack_bytes_per_s": hardware.inter_rack_bw,
            }
        )
    with (output_dir / "fixed_hardware_catalog.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(catalog_rows, handle, indent=2)
        handle.write("\n")

    rows = []
    for case in config["reference_cases"]:
        if case["catalog"] != "simulated":
            continue
        specification = DeploymentSpecification(
            model=models[case["model"]],
            seq_len=int(case["seq_len"]),
            tpot_slo_ms=float(case["tpot_slo_ms"]),
            budget=float(case["budget"]),
            hardware_menu=catalog,
        )
        result = provision_catalog(specification)
        plan = result.best_afd
        if plan is None:
            raise AssertionError(f"{case['name']} has no feasible AFD deployment")
        rows.append(
            {
                "case": case["name"],
                "model": case["model"],
                "seq_len": specification.seq_len,
                "tpot_slo_ms": specification.tpot_slo_ms,
                "budget": specification.budget,
                "selected_architecture": result.selected_architecture,
                "best_afd_attention_hardware": plan.attention_hardware,
                "best_afd_expert_hardware": plan.expert_hardware,
                "best_afd_throughput_tps": plan.throughput_tps,
                "best_collocated_hardware": result.best_collocated.attention_hardware,
                "best_collocated_throughput_tps": result.best_collocated.throughput_tps,
                "global_gain": result.comparison_gain,
            }
        )
    write_csv(output_dir / "simulated_catalog_reference.csv", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/simulated_catalog"),
    )
    args = parser.parse_args()
    rows = verify(args.output_dir)
    for row in rows:
        print(
            f"{row['case']}: "
            f"afd={row['best_afd_attention_hardware']}+"
            f"{row['best_afd_expert_hardware']} "
            f"collocated={row['best_collocated_hardware']} "
            f"G={row['global_gain']:.6f}"
        )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
