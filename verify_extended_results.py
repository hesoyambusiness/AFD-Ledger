#!/usr/bin/env python3
"""Generate extended fixed-catalog provisioning result artifacts."""

from __future__ import annotations

import argparse
import csv
import math
import platform
import time
from dataclasses import replace
from pathlib import Path


from sim.catalog_provisioning import (  # noqa: E402
    build_menu_frontiers,
    ordered_assignments,
    provision_catalog,
    provision_catalog_at_slo,
)
from sim.guided_catalog_search import (  # noqa: E402
    _attention_profile,
    _expert_profile,
    _pair_proxy,
    _role_diverse_beam,
    _role_anchors,
    _valid_assignments,
    guided_provision_catalog,
)
from sim.experiment_config import (  # noqa: E402
    load_experiment_config,
    load_experiment_models,
)
from sim.provisioning import (  # noqa: E402
    DeploymentSpecification,
    enumerate_afd,
    enumerate_collocated,
)
from sim.simulated_catalog import fixed_simulated_catalog  # noqa: E402
from sim.throughput_catalog import load_priced_existing_catalog  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def best_collocated(specification: DeploymentSpecification):
    return max(
        enumerate_collocated(specification),
        key=lambda plan: (plan.throughput_tps, -plan.tpot_ms),
        default=None,
    )


def replay_guided_search(
    plans,
    valid,
    attention_profiles,
    expert_profiles,
    *,
    evaluation_budget=20,
    beam_width=2,
    expansion_width=6,
    max_rounds=3,
):
    """Replay the outer policy against an already provisioned pair oracle."""

    valid_set = set(valid)
    budget = min(evaluation_budget, len(valid))
    anchors = {
        (attention, expert)
        for attention in _role_anchors(attention_profiles)
        for expert in _role_anchors(expert_profiles)
        if (attention, expert) in valid_set
    }
    proxy_order = sorted(
        valid,
        key=lambda pair: (
            _pair_proxy(pair, attention_profiles, expert_profiles), pair
        ),
        reverse=True,
    )
    evaluated = []

    def evaluate(pair):
        if pair not in evaluated and len(evaluated) < budget:
            evaluated.append(pair)

    for pair in sorted(
        anchors,
        key=lambda candidate: (
            _pair_proxy(candidate, attention_profiles, expert_profiles),
            candidate,
        ),
        reverse=True,
    ):
        evaluate(pair)
    for _ in range(max_rounds):
        feasible = [pair for pair in evaluated if pair in plans]
        if len(evaluated) >= budget or not feasible:
            break
        beam = _role_diverse_beam(
            {pair: plans[pair] for pair in feasible}, beam_width
        )
        neighbors = {
            pair for pair in valid
            if pair not in evaluated
            and any(pair[0] == current[0] or pair[1] == current[1] for current in beam)
        }
        ranked = sorted(
            neighbors,
            key=lambda pair: (
                _pair_proxy(pair, attention_profiles, expert_profiles),
                pair,
            ),
            reverse=True,
        )
        for pair in ranked[:expansion_width]:
            evaluate(pair)
    for pair in proxy_order:
        evaluate(pair)
    feasible = [pair for pair in evaluated if pair in plans]
    if not feasible:
        return None, tuple(evaluated)
    return (
        max(feasible, key=lambda pair: plans[pair].throughput_tps),
        tuple(evaluated),
    )


def generate_decision_sweep(models, existing) -> list[dict]:
    rows = []
    cases = (
        ("DeepSeek", "DeepSeek-V3.2", 4096, (32.0, 48.0, 64.0),
         (8.0, 10.0, 12.0, 14.0, 16.0, 20.0)),
        ("Qwen3", "Qwen3", 1024, (32.0, 48.0, 64.0),
         (6.0, 8.0, 10.0, 12.0, 15.0, 20.0)),
    )
    for label, model_name, seq_len, budgets, slo_grid in cases:
        for budget in budgets:
            specification = DeploymentSpecification(
                models[model_name], seq_len, max(slo_grid), budget, existing
            )
            frontiers = build_menu_frontiers(specification)
            for slo in slo_grid:
                at_slo = replace(specification, tpot_slo_ms=slo)
                collocated = best_collocated(at_slo)
                if collocated is None:
                    continue
                try:
                    result = provision_catalog_at_slo(frontiers, slo)
                    afd = result.best_afd
                    gain = result.comparison_gain
                    architecture = result.selected_architecture
                    attention_profiles = {
                        name: profile
                        for name, option in existing.items()
                        if (profile := _attention_profile(at_slo, option)) is not None
                    }
                    expert_profiles = {
                        name: profile
                        for name, option in existing.items()
                        if (profile := _expert_profile(at_slo, option)) is not None
                    }
                    valid = _valid_assignments(
                        at_slo, attention_profiles, expert_profiles
                    )
                    guided_pair, guided_order = replay_guided_search(
                        result.afd_plans,
                        valid,
                        attention_profiles,
                        expert_profiles,
                    )
                    proxy = max(
                        valid,
                        key=lambda pair: (
                            _pair_proxy(pair, attention_profiles, expert_profiles), pair
                        ),
                    )
                    local_pair, local_order = single_anchor_local(
                        proxy,
                        valid,
                        result.afd_plans,
                        attention_profiles,
                        expert_profiles,
                    )
                    guided_fraction = (
                        result.afd_plans[guided_pair].throughput_tps
                        / afd.throughput_tps if guided_pair else 0.0
                    )
                    local_fraction = (
                        result.afd_plans[local_pair].throughput_tps
                        / afd.throughput_tps if local_pair in result.afd_plans else 0.0
                    )
                    guided_first_best = (
                        guided_order.index(result.best_afd_assignment) + 1
                        if result.best_afd_assignment in guided_order else ""
                    )
                    local_evaluations = len(local_order)
                except ValueError:
                    afd = None
                    gain = 0.0
                    architecture = "collocated"
                    guided_fraction = 0.0
                    local_fraction = 0.0
                    guided_first_best = ""
                    local_evaluations = 0
                    proxy = None
                    local_pair = None
                    local_order = ()
                rows.append(
                    {
                        "model": label,
                        "seq_len": seq_len,
                        "budget": budget,
                        "tpot_slo_ms": slo,
                        "selected_architecture": architecture,
                        "global_gain": gain,
                        "best_afd_throughput_tps": (
                            afd.throughput_tps if afd is not None else 0.0
                        ),
                        "best_collocated_throughput_tps": collocated.throughput_tps,
                        "best_afd_pair": (
                            f"{afd.attention_hardware}+{afd.expert_hardware}"
                            if afd is not None else ""
                        ),
                        "best_collocated_hardware": collocated.attention_hardware,
                        "guided_fraction_of_oracle": guided_fraction,
                        "guided_first_best_attempt": guided_first_best,
                        "single_anchor_fraction_of_oracle": local_fraction,
                        "single_anchor_evaluations": local_evaluations,
                        "single_anchor_seed": (
                            f"{proxy[0]}+{proxy[1]}" if proxy else ""
                        ),
                        "single_anchor_selected_pair": (
                            f"{local_pair[0]}+{local_pair[1]}"
                            if local_pair else ""
                        ),
                        "single_anchor_attempted_pairs": ";".join(
                            f"{pair[0]}+{pair[1]}" for pair in local_order
                        ),
                    }
                )
    return rows


def harmonic(left: float, right: float) -> float:
    return 2.0 / (1.0 / left + 1.0 / right)


def component_pair(valid, catalog, attention_profiles, expert_profiles):
    attention_capacity = {
        name: catalog[name].hardware.hbm_capacity / catalog[name].price
        for name in attention_profiles
    }
    attention_bandwidth = {
        name: catalog[name].hardware.hbm_bw / catalog[name].price
        for name in attention_profiles
    }
    max_capacity = max(attention_capacity.values())
    max_bandwidth = max(attention_bandwidth.values())

    def fp8_per_price(name):
        peaks = catalog[name].hardware.peak_flops
        return peaks.get("fp8", peaks.get("int8", peaks["bf16"])) / catalog[name].price

    max_compute = max(fp8_per_price(name) for name in expert_profiles)
    return max(
        valid,
        key=lambda pair: (
            math.sqrt(
                attention_capacity[pair[0]] / max_capacity
                * attention_bandwidth[pair[0]] / max_bandwidth
            )
            * fp8_per_price(pair[1]) / max_compute,
            pair,
        ),
    )


def single_anchor_local(
    seed,
    valid,
    plans,
    attention_profiles,
    expert_profiles,
    *,
    evaluation_budget=20,
    expansion_width=6,
    max_rounds=3,
):
    """Ablation with one proxy seed, one retained pair, and no fallback."""

    evaluated = []

    def evaluate(pair):
        if pair not in evaluated and len(evaluated) < evaluation_budget:
            evaluated.append(pair)

    evaluate(seed)
    for _ in range(max_rounds):
        feasible = [pair for pair in evaluated if pair in plans]
        anchor = (
            max(feasible, key=lambda pair: plans[pair].throughput_tps)
            if feasible else seed
        )
        neighbors = [
            pair for pair in valid
            if pair not in evaluated
            and (pair[0] == anchor[0] or pair[1] == anchor[1])
        ]
        neighbors.sort(
            key=lambda pair: (
                _pair_proxy(pair, attention_profiles, expert_profiles), pair
            ),
            reverse=True,
        )
        for pair in neighbors[:expansion_width]:
            evaluate(pair)
    feasible = [pair for pair in evaluated if pair in plans]
    if not feasible:
        return seed, tuple(evaluated)
    return (
        max(feasible, key=lambda pair: plans[pair].throughput_tps),
        tuple(evaluated),
    )


def baseline_cases(models, existing, simulated):
    return (
        ("existing_deepseek", "DeepSeek-V3.2", existing, 4096, 12.0, 48.0),
        ("existing_qwen", "Qwen3", existing, 1024, 8.0, 64.0),
        ("simulated_deepseek_8ms", "DeepSeek-V3.2", simulated, 4096, 8.0, 48.0),
        ("simulated_deepseek_12ms", "DeepSeek-V3.2", simulated, 4096, 12.0, 48.0),
        ("simulated_qwen_10ms", "Qwen3", simulated, 1024, 10.0, 64.0),
    )


def generate_baselines(models, existing, simulated):
    rows = []
    landscape = []
    for label, model_name, catalog, seq_len, slo, budget in baseline_cases(
        models, existing, simulated
    ):
        specification = DeploymentSpecification(
            models[model_name], seq_len, slo, budget, catalog
        )
        oracle = provision_catalog(specification)
        guided = guided_provision_catalog(specification, evaluation_budget=20)
        attention_profiles = {
            name: profile
            for name, option in catalog.items()
            if (profile := _attention_profile(specification, option)) is not None
        }
        expert_profiles = {
            name: profile
            for name, option in catalog.items()
            if (profile := _expert_profile(specification, option)) is not None
        }
        valid = _valid_assignments(
            specification, attention_profiles, expert_profiles
        )
        cheapest = min(
            valid,
            key=lambda pair: (catalog[pair[0]].price + catalog[pair[1]].price, pair),
        )
        fastest = max(
            valid,
            key=lambda pair: (
                harmonic(
                    attention_profiles[pair[0]].service_rate,
                    expert_profiles[pair[1]].service_rate,
                ),
                pair,
            ),
        )
        component = component_pair(
            valid, catalog, attention_profiles, expert_profiles
        )
        proxy = max(
            valid,
            key=lambda pair: (
                _pair_proxy(pair, attention_profiles, expert_profiles), pair
            ),
        )
        local, local_order = single_anchor_local(
            proxy,
            valid,
            oracle.afd_plans,
            attention_profiles,
            expert_profiles,
        )
        policies = (
            ("cheapest_pair", cheapest, 1),
            ("fastest_roles", fastest, 1),
            ("component_rule", component, 1),
            ("profile_proxy", proxy, 1),
            ("single_anchor_local", local, len(local_order)),
            (
                "afd_ledger_20",
                (guided.best_afd.attention_hardware, guided.best_afd.expert_hardware),
                guided.evaluated_assignments,
            ),
            ("exhaustive_oracle", oracle.best_afd_assignment, len(valid)),
        )
        for policy, pair, evaluations in policies:
            plan = oracle.afd_plans.get(pair)
            throughput = plan.throughput_tps if plan is not None else 0.0
            rows.append(
                {
                    "case": label,
                    "policy": policy,
                    "attention_hardware": pair[0],
                    "expert_hardware": pair[1],
                    "throughput_tps": throughput,
                    "fraction_of_oracle": throughput / oracle.best_afd.throughput_tps,
                    "global_gain": throughput / oracle.best_collocated.throughput_tps,
                    "is_oracle_pair": pair == oracle.best_afd_assignment,
                    "pair_evaluations": evaluations,
                }
            )

        if label == "existing_deepseek":
            attempts = {
                pair: index
                for index, pair in enumerate(guided.evaluation_order, start=1)
            }
            for pair in sorted(valid):
                plan = oracle.afd_plans.get(pair)
                landscape.append(
                    {
                        "attention_hardware": pair[0],
                        "expert_hardware": pair[1],
                        "throughput_tps": plan.throughput_tps if plan else 0.0,
                        "fraction_of_oracle": (
                            plan.throughput_tps / oracle.best_afd.throughput_tps
                            if plan else 0.0
                        ),
                        "attempt": attempts.get(pair, ""),
                        "is_oracle_pair": pair == oracle.best_afd_assignment,
                        "feasible": plan is not None,
                    }
                )
    return rows, landscape


def scaled_option(option, factor: str, multiplier: float):
    if factor == "price":
        return replace(option, price=option.price * multiplier)
    hardware = option.hardware
    if factor == "capacity":
        hardware = replace(hardware, hbm_capacity=hardware.hbm_capacity * multiplier)
    elif factor == "hbm_bandwidth":
        hardware = replace(hardware, hbm_bw=hardware.hbm_bw * multiplier)
    elif factor == "compute":
        hardware = replace(
            hardware,
            peak_flops={
                precision: peak * multiplier
                for precision, peak in hardware.peak_flops.items()
            },
        )
    elif factor == "network":
        hardware = replace(
            hardware,
            intra_node_bw=hardware.intra_node_bw * multiplier,
            inter_node_bw=hardware.inter_node_bw * multiplier,
            inter_rack_bw=(
                hardware.inter_rack_bw * multiplier
                if hardware.inter_rack_bw is not None else None
            ),
        )
    else:
        raise ValueError(f"unknown sensitivity factor {factor}")
    return replace(option, hardware=hardware)


def generate_hardware_sensitivity(models, simulated):
    specification = DeploymentSpecification(
        models["DeepSeek-V3.2"], 4096, 12.0, 48.0, simulated
    )
    base = provision_catalog(specification)
    rows = []
    multipliers = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    sweeps = (
        ("attention", "sim_attention_target", "price"),
        ("attention", "sim_attention_target", "capacity"),
        ("attention", "sim_attention_target", "hbm_bandwidth"),
        ("expert", "sim_expert_target", "price"),
        ("expert", "sim_expert_target", "hbm_bandwidth"),
        ("expert", "sim_expert_target", "compute"),
        ("expert", "sim_expert_target", "network"),
    )
    for role, target, factor in sweeps:
        if role == "attention":
            affected = tuple(
                (target, expert)
                for expert, option in simulated.items()
                if expert != target and option.supports_expert
            )
            unaffected = [plan for pair, plan in base.afd_plans.items() if pair[0] != target]
        else:
            affected = tuple(
                (attention, target)
                for attention, option in simulated.items()
                if attention != target and option.supports_attention
            )
            unaffected = [plan for pair, plan in base.afd_plans.items() if pair[1] != target]
        for multiplier in multipliers:
            catalog = dict(simulated)
            catalog[target] = scaled_option(catalog[target], factor, multiplier)
            modified_specification = replace(specification, hardware_menu=catalog)
            candidates = list(unaffected)
            candidates.extend(enumerate_afd(modified_specification, affected))
            best = max(
                candidates,
                key=lambda plan: (plan.throughput_tps, -plan.tpot_ms),
            )
            rows.append(
                {
                    "role": role,
                    "factor": factor,
                    "multiplier": multiplier,
                    "best_afd_throughput_tps": best.throughput_tps,
                    "best_collocated_throughput_tps": base.best_collocated.throughput_tps,
                    "global_gain": best.throughput_tps / base.best_collocated.throughput_tps,
                    "best_pair": f"{best.attention_hardware}+{best.expert_hardware}",
                    "workers_attention": best.workers_attention,
                    "workers_expert": best.workers_expert,
                    "replicas": best.replicas,
                    "batch": best.batch_per_attention_card,
                    "pipeline_depth": best.pipeline_depth,
                }
            )
    return rows


def generate_overhead_sensitivity(models, existing, simulated):
    rows = []
    cases = (
        ("existing_deepseek", "DeepSeek-V3.2", existing, 4096, 12.0, 48.0),
        ("existing_qwen", "Qwen3", existing, 1024, 8.0, 64.0),
    )
    for label, model_name, catalog, seq_len, slo, budget in cases:
        specification = DeploymentSpecification(
            models[model_name], seq_len, slo, budget, catalog
        )
        collocated = best_collocated(specification)
        afd_plans = tuple(
            enumerate_afd(specification, ordered_assignments(catalog))
        )
        for overhead in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            feasible = [
                plan for plan in afd_plans
                if plan.tpot_ms * (1.0 + overhead) <= slo
            ]
            if feasible:
                best = max(
                    feasible,
                    key=lambda plan: (
                        plan.throughput_tps / (1.0 + overhead),
                        -plan.tpot_ms,
                    ),
                )
                throughput = best.throughput_tps / (1.0 + overhead)
                pair = f"{best.attention_hardware}+{best.expert_hardware}"
            else:
                throughput = 0.0
                pair = ""
            rows.append(
                {
                    "case": label,
                    "afd_runtime_overhead": overhead,
                    "best_afd_throughput_tps": throughput,
                    "best_collocated_throughput_tps": collocated.throughput_tps,
                    "global_gain": throughput / collocated.throughput_tps,
                    "selected_architecture": (
                        "afd"
                        if throughput > collocated.throughput_tps
                        else "collocated"
                    ),
                    "best_pair": pair,
                }
            )
    return rows


def generate_search_timing(models, existing, simulated):
    summaries = []
    pair_rows = []
    environment = f"{platform.platform()}|Python {platform.python_version()}"
    for label, model_name, catalog, seq_len, slo, budget in baseline_cases(
        models, existing, simulated
    ):
        if not label.startswith("existing_"):
            continue
        specification = DeploymentSpecification(
            models[model_name], seq_len, slo, budget, catalog
        )

        start = time.perf_counter()
        collocated_plans = tuple(enumerate_collocated(specification))
        collocated_seconds = time.perf_counter() - start
        if not collocated_plans:
            raise ValueError(f"{label}: no collocated plan")

        start = time.perf_counter()
        attention_profiles = {
            name: profile
            for name, option in catalog.items()
            if (profile := _attention_profile(specification, option)) is not None
        }
        expert_profiles = {
            name: profile
            for name, option in catalog.items()
            if (profile := _expert_profile(specification, option)) is not None
        }
        profile_seconds = time.perf_counter() - start
        valid = _valid_assignments(
            specification, attention_profiles, expert_profiles
        )

        plans = {}
        pair_seconds = {}
        for pair in valid:
            paired = specification.with_menu(pair)
            start = time.perf_counter()
            candidates = tuple(enumerate_afd(paired, (pair,)))
            elapsed = time.perf_counter() - start
            pair_seconds[pair] = elapsed
            if candidates:
                plans[pair] = max(
                    candidates,
                    key=lambda plan: (plan.throughput_tps, -plan.tpot_ms),
                )
            pair_rows.append(
                {
                    "case": label,
                    "attention_hardware": pair[0],
                    "expert_hardware": pair[1],
                    "pair_search_seconds": elapsed,
                    "feasible": bool(candidates),
                    "environment": environment,
                }
            )

        start = time.perf_counter()
        guided_pair, guided_order = replay_guided_search(
            plans, valid, attention_profiles, expert_profiles
        )
        controller_seconds = time.perf_counter() - start
        if guided_pair is None:
            raise ValueError(f"{label}: bounded search found no feasible pair")
        bounded_pair_seconds = sum(pair_seconds[pair] for pair in guided_order)
        exhaustive_pair_seconds = sum(pair_seconds.values())
        bounded_total = (
            collocated_seconds + profile_seconds
            + controller_seconds + bounded_pair_seconds
        )
        exhaustive_total = collocated_seconds + exhaustive_pair_seconds
        summaries.append(
            {
                "case": label,
                "valid_pairs": len(valid),
                "bounded_pairs": len(guided_order),
                "pair_fraction": len(guided_order) / len(valid),
                "collocated_seconds": collocated_seconds,
                "profile_seconds": profile_seconds,
                "controller_seconds": controller_seconds,
                "bounded_pair_seconds": bounded_pair_seconds,
                "exhaustive_pair_seconds": exhaustive_pair_seconds,
                "bounded_total_seconds": bounded_total,
                "exhaustive_total_seconds": exhaustive_total,
                "time_fraction": bounded_total / exhaustive_total,
                "search_speedup": exhaustive_total / bounded_total,
                "environment": environment,
            }
        )
    return summaries, pair_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/extended_results"),
    )
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "evaluation",
            "decision",
            "baselines",
            "hardware",
            "overhead",
            "timing",
        ),
        default="all",
    )
    args = parser.parse_args()

    config = load_experiment_config()
    models = load_experiment_models(config)
    commercial = dict(load_priced_existing_catalog(config))
    existing_names = config["existing_catalog"]["constructed_availability_subset"]
    existing = {name: commercial[name] for name in existing_names}
    simulated = dict(
        fixed_simulated_catalog(commercial, config["simulated_catalog"])
    )

    if args.only in {"all", "evaluation", "decision"}:
        decision_rows = generate_decision_sweep(models, existing)
        write_csv(args.output_dir / "existing_decision_sweep.csv", decision_rows)

    if args.only in {"all", "evaluation", "baselines"}:
        baseline_rows, landscape_rows = generate_baselines(models, existing, simulated)
        write_csv(args.output_dir / "pair_heuristic_baselines.csv", baseline_rows)
        if args.only in {"all", "baselines"}:
            write_csv(args.output_dir / "deepseek_pair_landscape.csv", landscape_rows)

    if args.only in {"all", "evaluation", "hardware"}:
        hardware_rows = generate_hardware_sensitivity(models, simulated)
        write_csv(args.output_dir / "simulated_hardware_sensitivity.csv", hardware_rows)

    if args.only in {"all", "evaluation", "overhead"}:
        overhead_rows = generate_overhead_sensitivity(models, existing, simulated)
        write_csv(args.output_dir / "runtime_overhead_sensitivity.csv", overhead_rows)

    if args.only in {"all", "timing"}:
        timing_rows, pair_timing_rows = generate_search_timing(
            models, existing, simulated
        )
        write_csv(args.output_dir / "search_time_summary.csv", timing_rows)
        write_csv(args.output_dir / "pair_search_times.csv", pair_timing_rows)
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
