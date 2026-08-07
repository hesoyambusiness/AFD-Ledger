#!/usr/bin/env python3
"""Verify seed-free catalog provisioning across TPOT SLOs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sim.catalog_provisioning import (
    build_menu_frontiers,
    provision_catalog_at_slo,
)
from sim.experiment_config import load_experiment_config, load_experiment_models
from sim.provisioning import DeploymentSpecification
from sim.simulated_catalog import fixed_simulated_catalog
from sim.throughput_catalog import load_priced_existing_catalog


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify(output_dir: Path, include_simulated_catalog: bool = True) -> list[dict]:
    config = load_experiment_config()
    models = load_experiment_models(config)
    names = tuple(config["simulated_catalog"]["general_devices"])
    catalog = dict(load_priced_existing_catalog(config, names))
    if include_simulated_catalog:
        catalog = dict(fixed_simulated_catalog(catalog, config["simulated_catalog"]))
    summary_rows = []
    menu_rows = []

    for label, case in config["slo_sweeps"].items():
        slo_grid = tuple(float(value) for value in case["slo_grid_ms"])
        specification = DeploymentSpecification(
            model=models[case["model"]],
            seq_len=int(case["seq_len"]),
            tpot_slo_ms=max(slo_grid),
            budget=float(case["budget"]),
            hardware_menu=catalog,
        )
        frontiers = build_menu_frontiers(specification)
        for slo in slo_grid:
            result = provision_catalog_at_slo(frontiers, slo)
            summary_rows.append(
                {
                    "model_label": label,
                    "tpot_slo_ms": slo,
                    "selected_architecture": result.selected_architecture,
                    "best_attention_hardware": result.best_afd_assignment[0],
                    "best_expert_hardware": result.best_afd_assignment[1],
                    "global_comparison_gain": result.comparison_gain,
                    "best_afd_throughput_tps": result.best_afd.throughput_tps,
                    "best_collocated_hardware": result.best_collocated.attention_hardware,
                    "best_collocated_throughput_tps": result.best_collocated.throughput_tps,
                    "feasible_afd_menus": len(result.afd_plans),
                    "feasible_collocated_devices": len(result.collocated_plans),
                }
            )
            for assignment, plan in sorted(result.afd_plans.items()):
                menu_rows.append(
                    {
                        "model_label": label,
                        "tpot_slo_ms": slo,
                        "attention_hardware": assignment[0],
                        "expert_hardware": assignment[1],
                        "afd_throughput_tps": plan.throughput_tps,
                        "fraction_of_best_afd": (
                            plan.throughput_tps / result.best_afd.throughput_tps
                        ),
                        "is_best_afd_menu": assignment == result.best_afd_assignment,
                        "best_collocated_throughput_tps": (
                            result.best_collocated.throughput_tps
                        ),
                    }
                )

    write_csv(output_dir / "catalog_slo_summary.csv", summary_rows)
    write_csv(output_dir / "catalog_menu_landscape.csv", menu_rows)
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/throughput_catalog_slo_sweep"),
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="exclude the simulated role-specialized catalog entries",
    )
    args = parser.parse_args()
    rows = verify(
        args.output_dir,
        include_simulated_catalog=not args.existing_only,
    )
    for row in rows:
        print(
            f"{row['model_label']} tau={row['tpot_slo_ms']:.1f}ms "
            f"selected={row['selected_architecture']} "
            f"afd={row['best_attention_hardware']}+{row['best_expert_hardware']} "
            f"collocated={row['best_collocated_hardware']} "
            f"G={row['global_comparison_gain']:.6f}"
        )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
