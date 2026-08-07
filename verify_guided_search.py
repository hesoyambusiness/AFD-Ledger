#!/usr/bin/env python3
"""Compare bounded fixed-catalog search with exact pair oracles."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from sim.catalog_provisioning import provision_catalog
from sim.experiment_config import load_experiment_config, load_experiment_models
from sim.guided_catalog_search import guided_provision_catalog
from sim.provisioning import DeploymentSpecification
from sim.simulated_catalog import fixed_simulated_catalog
from sim.throughput_catalog import load_priced_existing_catalog


def component_efficiency_heuristic(catalog):
    """Select attention by HBM/$ and experts by supported 8-bit compute/$.

    The commercial catalog has one device that independently maximizes both
    attention-side signals, so this rule does not require an arbitrary weight
    between memory capacity and bandwidth.
    """

    attention_by_capacity = max(
        catalog,
        key=lambda name: (
            catalog[name].hardware.hbm_capacity / catalog[name].price,
            name,
        ),
    )
    attention_by_bandwidth = max(
        catalog,
        key=lambda name: (
            catalog[name].hardware.hbm_bw / catalog[name].price,
            name,
        ),
    )
    if attention_by_capacity != attention_by_bandwidth:
        raise ValueError(
            "attention heuristic has no device maximizing both HBM signals"
        )

    def eight_bit_peak(name):
        peaks = catalog[name].hardware.peak_flops
        return peaks.get("fp8", peaks.get("int8", -math.inf))

    expert = max(
        catalog,
        key=lambda name: (eight_bit_peak(name) / catalog[name].price, name),
    )
    return attention_by_capacity, expert


def verify(output_dir: Path, evaluation_budget: int) -> list[dict]:
    config = load_experiment_config()
    models = load_experiment_models(config)
    commercial = load_priced_existing_catalog(config)
    subset_names = tuple(
        config["existing_catalog"]["constructed_availability_subset"]
    )
    existing = {name: commercial[name] for name in subset_names}
    simulated = dict(
        fixed_simulated_catalog(commercial, config["simulated_catalog"])
    )
    catalogs = {
        "constructed_existing": existing,
        "simulated": simulated,
    }
    rows = []
    trace_rows = []
    for case in config["reference_cases"]:
        label = str(case["name"])
        model_name = str(case["model"])
        catalog = catalogs[str(case["catalog"])]
        seq_len = int(case["seq_len"])
        tpot_ms = float(case["tpot_slo_ms"])
        budget = float(case["budget"])
        specification = DeploymentSpecification(
            models[model_name],
            seq_len,
            tpot_ms,
            budget,
            catalog,
        )
        guided = guided_provision_catalog(
            specification,
            evaluation_budget=evaluation_budget,
        )
        oracle = provision_catalog(specification)
        heuristic_assignment = (
            component_efficiency_heuristic(catalog)
            if label.startswith("existing_")
            else None
        )
        heuristic_plan = (
            oracle.afd_plans.get(heuristic_assignment)
            if heuristic_assignment is not None
            else None
        )
        guided_assignment = (
            guided.best_afd.attention_hardware,
            guided.best_afd.expert_hardware,
        )
        throughput_regret = (
            oracle.best_afd.throughput_tps - guided.best_afd.throughput_tps
        ) / oracle.best_afd.throughput_tps
        rows.append(
            {
                "case": label,
                "model": model_name,
                "seq_len": seq_len,
                "tpot_slo_ms": tpot_ms,
                "budget": budget,
                "catalog_size": len(catalog),
                "valid_ordered_pairs": guided.valid_assignments,
                "evaluated_ordered_pairs": guided.evaluated_assignments,
                "feasible_evaluated_pairs": guided.feasible_evaluated_assignments,
                "evaluation_fraction": (
                    guided.evaluated_assignments / guided.valid_assignments
                ),
                "guided_attention_hardware": guided_assignment[0],
                "guided_expert_hardware": guided_assignment[1],
                "oracle_attention_hardware": oracle.best_afd_assignment[0],
                "oracle_expert_hardware": oracle.best_afd_assignment[1],
                "oracle_collocated_hardware": (
                    oracle.best_collocated.attention_hardware
                ),
                "guided_afd_throughput_tps": guided.best_afd.throughput_tps,
                "oracle_afd_throughput_tps": oracle.best_afd.throughput_tps,
                "oracle_collocated_throughput_tps": oracle.best_collocated.throughput_tps,
                "oracle_global_gain": oracle.comparison_gain,
                "heuristic_attention_hardware": (
                    heuristic_assignment[0] if heuristic_assignment else ""
                ),
                "heuristic_expert_hardware": (
                    heuristic_assignment[1] if heuristic_assignment else ""
                ),
                "heuristic_afd_throughput_tps": (
                    heuristic_plan.throughput_tps if heuristic_plan else 0.0
                ),
                "heuristic_fraction_of_oracle": (
                    heuristic_plan.throughput_tps / oracle.best_afd.throughput_tps
                    if heuristic_plan
                    else 0.0
                ),
                "throughput_regret": throughput_regret,
                "matches_oracle_pair": guided_assignment
                == oracle.best_afd_assignment,
                "evaluation_order": " -> ".join(
                    f"{attention}+{expert}"
                    for attention, expert in guided.evaluation_order
                ),
            }
        )
        best_so_far = 0.0
        for attempt, assignment in enumerate(guided.evaluation_order, start=1):
            plan = guided.evaluated_plans.get(assignment)
            throughput = plan.throughput_tps if plan is not None else 0.0
            best_so_far = max(best_so_far, throughput)
            trace_rows.append(
                {
                    "case": label,
                    "attempt": attempt,
                    "attention_hardware": assignment[0],
                    "expert_hardware": assignment[1],
                    "feasible": plan is not None,
                    "throughput_tps": throughput,
                    "best_so_far_tps": best_so_far,
                    "best_so_far_fraction": (
                        best_so_far / oracle.best_afd.throughput_tps
                    ),
                    "is_oracle_pair": assignment == oracle.best_afd_assignment,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with (output_dir / "guided_search_convergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    trace_fields = sorted({field for row in trace_rows for field in row})
    with (output_dir / "guided_search_trace.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        writer.writeheader()
        writer.writerows(trace_rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/guided_search"),
    )
    parser.add_argument("--evaluation-budget", type=int, default=20)
    args = parser.parse_args()
    rows = verify(args.output_dir, args.evaluation_budget)
    for row in rows:
        print(
            f"{row['case']}: pair={row['guided_attention_hardware']}+"
            f"{row['guided_expert_hardware']} "
            f"evaluated={row['evaluated_ordered_pairs']}/"
            f"{row['valid_ordered_pairs']} "
            f"regret={row['throughput_regret']:.6f} "
            f"G={row['oracle_global_gain']:.6f}"
        )
    if any(row["throughput_regret"] > 1e-12 for row in rows):
        return 1
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
