#!/usr/bin/env python3
"""Evaluate homogeneous AFD against collocated serving on the paper grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from sim.experiment_config import load_experiment_config, load_experiment_models
from sim.precision import resolve_compute_precision
from sim.provisioning import (
    DeploymentPlan,
    DeploymentSpecification,
    enumerate_afd,
    enumerate_collocated,
)
from sim.throughput_catalog import load_priced_existing_catalog


CASES = (
    (
        "DeepSeek",
        "DeepSeek-V3.2",
        4096,
        (32.0, 48.0, 64.0),
        (8.0, 10.0, 12.0, 14.0, 16.0, 20.0),
    ),
    (
        "Qwen3",
        "Qwen3",
        1024,
        (32.0, 48.0, 64.0),
        (6.0, 8.0, 10.0, 12.0, 15.0, 20.0),
    ),
)


def best_at_slo(
    plans: tuple[DeploymentPlan, ...],
    tpot_slo_ms: float,
) -> DeploymentPlan | None:
    feasible = (plan for plan in plans if plan.tpot_ms <= tpot_slo_ms)
    return max(
        feasible,
        key=lambda plan: (plan.throughput_tps, -plan.tpot_ms),
        default=None,
    )


def ffn_mfu(model, plan: DeploymentPlan, hardware, *, afd: bool) -> float:
    if afd:
        routed_tokens = math.ceil(
            model.top_k
            * plan.workers_attention
            * plan.microbatch_size
            / plan.workers_expert
        )
    else:
        routed_tokens = model.top_k * plan.microbatch_size
    precision = resolve_compute_precision(
        model.compute_precision,
        hardware.peak_flops,
    )
    peak_flops = hardware.peak_flops[precision]
    useful_flops = 6.0 * routed_tokens * model.hidden * model.moe_hidden
    expert_layer_s = plan.stage_expert_ms / 1000.0 / model.layers
    return useful_flops / (expert_layer_s * peak_flops)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    config = load_experiment_config()
    models = load_experiment_models(config)
    commercial = load_priced_existing_catalog(config)
    names = tuple(config["existing_catalog"]["constructed_availability_subset"])
    catalog = {name: commercial[name] for name in names}
    global_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    for label, model_name, seq_len, budgets, slo_grid in CASES:
        model = models[model_name]
        for budget in budgets:
            specification = DeploymentSpecification(
                model=model,
                seq_len=seq_len,
                tpot_slo_ms=max(slo_grid),
                budget=budget,
                hardware_menu=catalog,
            )
            collocated_plans = tuple(enumerate_collocated(specification))
            homogeneous_plans = {
                name: tuple(enumerate_afd(specification, ((name, name),)))
                for name in names
            }
            collocated_by_hardware = {
                name: tuple(
                    plan
                    for plan in collocated_plans
                    if plan.attention_hardware == name
                )
                for name in names
            }

            for slo in slo_grid:
                best_collocated = best_at_slo(collocated_plans, slo)
                best_homogeneous_by_hardware = {
                    name: plan
                    for name in names
                    if (plan := best_at_slo(homogeneous_plans[name], slo))
                    is not None
                }
                if best_collocated is None or not best_homogeneous_by_hardware:
                    continue
                best_homogeneous = max(
                    best_homogeneous_by_hardware.values(),
                    key=lambda plan: (plan.throughput_tps, -plan.tpot_ms),
                )
                global_rows.append(
                    {
                        "model": label,
                        "seq_len": seq_len,
                        "budget": budget,
                        "tpot_slo_ms": slo,
                        "best_homogeneous_hardware": best_homogeneous.attention_hardware,
                        "best_homogeneous_throughput_tps": best_homogeneous.throughput_tps,
                        "best_homogeneous_tpot_ms": best_homogeneous.tpot_ms,
                        "best_homogeneous_concurrency": best_homogeneous.concurrency,
                        "best_collocated_hardware": best_collocated.attention_hardware,
                        "best_collocated_throughput_tps": best_collocated.throughput_tps,
                        "best_collocated_tpot_ms": best_collocated.tpot_ms,
                        "best_collocated_concurrency": best_collocated.concurrency,
                        "homogeneous_global_gain": (
                            best_homogeneous.throughput_tps
                            / best_collocated.throughput_tps
                        ),
                        "homogeneous_tpot_speedup": (
                            best_collocated.tpot_ms / best_homogeneous.tpot_ms
                        ),
                        "homogeneous_concurrency_ratio": (
                            best_homogeneous.concurrency
                            / best_collocated.concurrency
                        ),
                    }
                )

                for name, homogeneous in best_homogeneous_by_hardware.items():
                    collocated = best_at_slo(collocated_by_hardware[name], slo)
                    if collocated is None:
                        continue
                    option = catalog[name]
                    paired_rows.append(
                        {
                            "model": label,
                            "seq_len": seq_len,
                            "budget": budget,
                            "tpot_slo_ms": slo,
                            "hardware": name,
                            "homogeneous_throughput_tps": homogeneous.throughput_tps,
                            "collocated_throughput_tps": collocated.throughput_tps,
                            "throughput_ratio": (
                                homogeneous.throughput_tps
                                / collocated.throughput_tps
                            ),
                            "homogeneous_tpot_ms": homogeneous.tpot_ms,
                            "collocated_tpot_ms": collocated.tpot_ms,
                            "tpot_speedup": collocated.tpot_ms / homogeneous.tpot_ms,
                            "homogeneous_concurrency": homogeneous.concurrency,
                            "collocated_concurrency": collocated.concurrency,
                            "concurrency_ratio": (
                                homogeneous.concurrency / collocated.concurrency
                            ),
                            "homogeneous_attention_workers": (
                                homogeneous.workers_attention
                            ),
                            "homogeneous_expert_workers": homogeneous.workers_expert,
                            "homogeneous_replicas": homogeneous.replicas,
                            "homogeneous_batch_per_attention_card": (
                                homogeneous.batch_per_attention_card
                            ),
                            "homogeneous_microbatch_size": homogeneous.microbatch_size,
                            "homogeneous_pipeline_depth": homogeneous.pipeline_depth,
                            "homogeneous_communication_path": (
                                homogeneous.communication_path
                            ),
                            "collocated_ep_width": collocated.workers_expert,
                            "collocated_replicas": collocated.replicas,
                            "collocated_batch_per_card": (
                                collocated.batch_per_attention_card
                            ),
                            "collocated_microbatch_size": collocated.microbatch_size,
                            "collocated_pipeline_depth": collocated.pipeline_depth,
                            "collocated_communication_path": (
                                collocated.communication_path
                            ),
                            "homogeneous_ffn_mfu": ffn_mfu(
                                model,
                                homogeneous,
                                option.hardware,
                                afd=True,
                            ),
                            "collocated_ffn_mfu": ffn_mfu(
                                model,
                                collocated,
                                option.hardware,
                                afd=False,
                            ),
                        }
                    )
    return global_rows, paired_rows


def summarize(
    global_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
) -> dict[str, object]:
    global_wins = [row for row in global_rows if row["homogeneous_global_gain"] > 1.0]
    global_faster = [row for row in global_rows if row["homogeneous_tpot_speedup"] > 1.0]
    paired_wins = [row for row in paired_rows if row["throughput_ratio"] > 1.0]
    paired_faster = [row for row in paired_rows if row["tpot_speedup"] > 1.0]
    faster_but_loses = [
        row
        for row in paired_faster
        if row["throughput_ratio"] <= 1.0
    ]
    higher_mfu = [
        row
        for row in paired_rows
        if row["homogeneous_ffn_mfu"] > row["collocated_ffn_mfu"]
    ]
    higher_mfu_but_loses = [
        row
        for row in higher_mfu
        if row["throughput_ratio"] <= 1.0
    ]
    global_by_model = {}
    for model in ("DeepSeek", "Qwen3"):
        rows = [row for row in global_rows if row["model"] == model]
        global_by_model[model] = {
            "specifications": len(rows),
            "homogeneous_wins": sum(
                row["homogeneous_global_gain"] > 1.0 for row in rows
            ),
        }
    same_device_wins_by_model = {
        model: sum(row["model"] == model for row in paired_wins)
        for model in ("DeepSeek", "Qwen3")
    }
    global_winning_hardware = {
        hardware: sum(
            row["best_homogeneous_hardware"] == hardware
            for row in global_wins
        )
        for hardware in sorted(
            {row["best_homogeneous_hardware"] for row in global_wins}
        )
    }
    max_same_device_gain = max(
        paired_rows,
        key=lambda row: row["throughput_ratio"],
    )
    return {
        "global_specifications": len(global_rows),
        "global_homogeneous_wins": len(global_wins),
        "global_homogeneous_tpot_speedups": len(global_faster),
        "global_gain_min": min(row["homogeneous_global_gain"] for row in global_rows),
        "global_gain_max": max(row["homogeneous_global_gain"] for row in global_rows),
        "global_by_model": global_by_model,
        "global_winning_hardware": global_winning_hardware,
        "same_device_comparisons": len(paired_rows),
        "same_device_homogeneous_wins": len(paired_wins),
        "same_device_wins_by_model": same_device_wins_by_model,
        "same_device_wins_with_concurrency_gain": sum(
            row["concurrency_ratio"] > 1.0 for row in paired_wins
        ),
        "same_device_wins_with_tpot_speedup": sum(
            row["tpot_speedup"] > 1.0 for row in paired_wins
        ),
        "same_device_gain_max": max(
            row["throughput_ratio"] for row in paired_rows
        ),
        "same_device_gain_max_case": {
            "model": max_same_device_gain["model"],
            "budget": max_same_device_gain["budget"],
            "tpot_slo_ms": max_same_device_gain["tpot_slo_ms"],
            "hardware": max_same_device_gain["hardware"],
            "throughput_ratio": max_same_device_gain["throughput_ratio"],
            "concurrency_ratio": max_same_device_gain["concurrency_ratio"],
            "tpot_speedup": max_same_device_gain["tpot_speedup"],
        },
        "same_device_tpot_speedups": len(paired_faster),
        "same_device_tpot_speedups_with_throughput_loss": len(faster_but_loses),
        "same_device_higher_ffn_mfu": len(higher_mfu),
        "same_device_higher_ffn_mfu_with_throughput_loss": len(
            higher_mfu_but_loses
        ),
        "faster_but_losing_median_tpot_speedup": statistics.median(
            row["tpot_speedup"] for row in faster_but_loses
        ),
        "faster_but_losing_median_concurrency_ratio": statistics.median(
            row["concurrency_ratio"] for row in faster_but_loses
        ),
        "faster_but_losing_median_throughput_ratio": statistics.median(
            row["throughput_ratio"] for row in faster_but_loses
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/homogeneous_afd"),
    )
    args = parser.parse_args()
    global_rows, paired_rows = generate()
    summary = summarize(global_rows, paired_rows)
    write_csv(args.output_dir / "homogeneous_global_grid.csv", global_rows)
    write_csv(args.output_dir / "homogeneous_same_device_grid.csv", paired_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
