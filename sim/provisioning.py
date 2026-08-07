"""Comparison-aware deployment search used by AFD-Ledger.

This module implements the deployment specification, collocated/AFD search,
and execution-organization optimization used by AFD-Ledger. It deliberately
reuses the simulator's operator model so there is one source of truth for
latency and memory formulas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from .operators import (
    attention_layer_time,
    dispatch_or_combine_payloads,
    expert_layer_time,
    kv_cache_bytes,
    collocated_comm_time,
)
from .precision import resolve_compute_precision
from .schedules import afd_iteration_schedule
from .specs import Hardware, ModelSpec


DEFAULT_BATCHES = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


@dataclass(frozen=True)
class HardwareOption:
    """A hardware device together with its normalized hourly price."""

    name: str
    hardware: Hardware
    price: float
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.price <= 0 or not math.isfinite(self.price):
            raise ValueError(f"hardware {self.name}: price must be finite and positive")

    @property
    def supports_attention(self) -> bool:
        return "expert_only" not in self.tags

    @property
    def supports_expert(self) -> bool:
        return "attention_only" not in self.tags

    @property
    def supports_collocated(self) -> bool:
        return self.supports_attention and self.supports_expert


@dataclass(frozen=True)
class SearchConstraints:
    """Implementation constraints shared by both architectural searches."""

    max_microbatch: int = 128
    batch_candidates: Tuple[int, ...] = DEFAULT_BATCHES
    routing_tail_factor: float = 1.0
    communication_scale: float = 1.0
    exhaustive_allocations: bool = False
    exhaustive_batches: bool = False

    def __post_init__(self) -> None:
        if self.max_microbatch < 1:
            raise ValueError("max_microbatch must be positive")
        if self.routing_tail_factor < 1:
            raise ValueError("routing_tail_factor must be >= 1")
        if self.communication_scale < 0:
            raise ValueError("communication_scale must be non-negative")
        if not self.batch_candidates or any(value < 1 for value in self.batch_candidates):
            raise ValueError("batch_candidates must contain positive integers")


@dataclass(frozen=True)
class DeploymentSpecification:
    """Validated deployment contract shared by both architecture searches."""

    model: ModelSpec
    seq_len: int
    tpot_slo_ms: float
    budget: float
    hardware_menu: Mapping[str, HardwareOption]
    constraints: SearchConstraints = SearchConstraints()
    workload: str = "steady_state_decode"

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        if self.tpot_slo_ms <= 0 or not math.isfinite(self.tpot_slo_ms):
            raise ValueError("tpot_slo_ms must be finite and positive")
        if self.budget <= 0 or not math.isfinite(self.budget):
            raise ValueError("budget must be finite and positive")
        if not self.hardware_menu:
            raise ValueError("hardware_menu cannot be empty")
        object.__setattr__(
            self,
            "hardware_menu",
            MappingProxyType(dict(self.hardware_menu)),
        )
        for name, option in self.hardware_menu.items():
            if name != option.name:
                raise ValueError(f"hardware menu key {name!r} does not match option {option.name!r}")
            requested = {
                self.model.compute_precision,
                self.model.attention_precision,
                self.model.router_precision,
            }
            try:
                for precision in requested:
                    resolve_compute_precision(
                        precision,
                        option.hardware.peak_flops,
                    )
            except ValueError as error:
                raise ValueError(
                    f"hardware {name} cannot execute the requested arithmetic: {error}"
                ) from error

    def with_menu(self, names: Sequence[str]) -> "DeploymentSpecification":
        menu = {name: self.hardware_menu[name] for name in dict.fromkeys(names)}
        return replace(self, hardware_menu=menu)


@dataclass(frozen=True)
class DeploymentPlan:
    architecture: str
    attention_hardware: str
    expert_hardware: str
    workers_attention: int
    workers_expert: int
    replicas: int
    batch_per_attention_card: int
    microbatch_size: int
    pipeline_depth: int
    tpot_ms: float
    throughput_tps: float
    concurrency: int
    budget_used: float
    stage_attention_ms: float
    stage_dispatch_ms: float
    stage_expert_ms: float
    stage_combine_ms: float
    communication_path: str
    execution_organization: str

def _communication_path(total_cards: int, left: Hardware, right: Hardware) -> str:
    if total_cards <= min(left.gpus_per_node, right.gpus_per_node):
        return "intra"
    if left.gpus_per_rack is not None and right.gpus_per_rack is not None:
        if total_cards <= min(left.gpus_per_rack, right.gpus_per_rack):
            return "inter"
    return "rdma"


def _communication_bandwidth(path: str, left: Hardware, right: Hardware) -> float:
    return min(left.comm_bw(path), right.comm_bw(path))


def _batch_values(max_batch: int, limits: SearchConstraints, depth: int) -> Tuple[int, ...]:
    upper = min(max_batch, limits.max_microbatch * depth)
    if upper < 1:
        return ()
    if limits.exhaustive_batches:
        return tuple(range(1, upper + 1))
    values = {value for value in limits.batch_candidates if value <= upper}
    values.add(upper)
    return tuple(sorted(values))


def _collocated_plan(
    spec: DeploymentSpecification,
    option: HardwareOption,
    ep_width: int,
    batch_per_card: int,
    overlap_batches: int,
) -> Optional[DeploymentPlan]:
    model = spec.model
    hw = option.hardware
    cards = int(math.floor((spec.budget + 1e-12) / option.price))
    if ep_width < 1 or ep_width > min(model.experts, cards):
        return None
    replicas = cards // ep_width
    if replicas < 1:
        return None

    fixed_memory = model.attn_weight_bytes + model.moe_weight_bytes_total / ep_width
    memory = fixed_memory + kv_cache_bytes(model, batch_per_card, spec.seq_len)
    if memory > hw.hbm_capacity:
        return None

    microbatch = math.ceil(batch_per_card / overlap_batches)
    if microbatch > spec.constraints.max_microbatch:
        return None
    path = "intra" if ep_width <= hw.gpus_per_node else "inter"
    attention = attention_layer_time(model, hw, microbatch, spec.seq_len) * model.layers
    expert = (
        expert_layer_time(model, hw, model.top_k * microbatch, ep_width)
        * model.layers
        * spec.constraints.routing_tail_factor
    )
    communication = (
        collocated_comm_time(model, hw, ep_width, microbatch, path)
        * model.layers
        * spec.constraints.communication_scale
    )
    if overlap_batches == 1:
        cycle = attention + expert + 2.0 * communication
        organization = "sequential"
    else:
        cycle = (
            max(attention, communication)
            + max(communication, attention)
            + max(expert, communication)
            + max(communication, expert)
        )
        organization = "two_batch_overlap"

    concurrency = replicas * ep_width * batch_per_card
    return DeploymentPlan(
        architecture="collocated",
        attention_hardware=option.name,
        expert_hardware=option.name,
        workers_attention=ep_width,
        workers_expert=ep_width,
        replicas=replicas,
        batch_per_attention_card=batch_per_card,
        microbatch_size=microbatch,
        pipeline_depth=overlap_batches,
        tpot_ms=cycle * 1e3,
        throughput_tps=concurrency / cycle,
        concurrency=concurrency,
        budget_used=cards * option.price,
        stage_attention_ms=attention * 1e3,
        stage_dispatch_ms=communication * 1e3,
        stage_expert_ms=expert * 1e3,
        stage_combine_ms=communication * 1e3,
        communication_path=path,
        execution_organization=organization,
    )


def enumerate_collocated(spec: DeploymentSpecification) -> Iterator[DeploymentPlan]:
    """Enumerate every SLO-feasible collocated organization in the menu."""

    model = spec.model
    for option in spec.hardware_menu.values():
        if not option.supports_collocated:
            continue
        hw = option.hardware
        cards = int(math.floor((spec.budget + 1e-12) / option.price))
        for ep_width in range(1, min(model.experts, cards) + 1):
            fixed = model.attn_weight_bytes + model.moe_weight_bytes_total / ep_width
            kv_room = hw.hbm_capacity - fixed
            if kv_room <= 0:
                continue
            max_batch = int(kv_room // kv_cache_bytes(model, 1, spec.seq_len))
            for overlap_batches in (1, 2):
                for batch in _batch_values(max_batch, spec.constraints, overlap_batches):
                    plan = _collocated_plan(
                        spec, option, ep_width, batch, overlap_batches
                    )
                    if plan is not None and plan.tpot_ms <= spec.tpot_slo_ms:
                        yield plan


def _allocation_values(
    spec: DeploymentSpecification,
    attention: HardwareOption,
    expert: HardwareOption,
) -> Iterator[Tuple[int, int]]:
    model = spec.model
    min_expert = max(1, math.ceil(model.moe_weight_bytes_total / expert.hardware.hbm_capacity))
    max_expert = min(
        model.experts,
        int((spec.budget - attention.price + 1e-12) // expert.price),
    )
    if min_expert > max_expert:
        return

    if spec.constraints.exhaustive_allocations:
        expert_values: Iterable[int] = range(min_expert, max_expert + 1)
    else:
        values = set(range(min_expert, min(max_expert, min_expert + 8) + 1))
        values.update(x for x in range(1, max_expert + 1) if model.experts % x == 0)
        values.update((8, 16, 24, 32, 48, 64, 96, 128, 192, 256, max_expert))
        expert_values = sorted(x for x in values if min_expert <= x <= max_expert)

    for expert_workers in expert_values:
        if model.moe_weight_bytes_total / expert_workers > expert.hardware.hbm_capacity:
            continue
        max_attention = int(
            (spec.budget - expert_workers * expert.price + 1e-12) // attention.price
        )
        if max_attention < 1:
            continue
        if spec.constraints.exhaustive_allocations:
            attention_values: Iterable[int] = range(1, max_attention + 1)
        else:
            values = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, max_attention}
            values.update((max(1, max_attention - 1), max(1, max_attention - 2)))
            for fraction in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
                values.add(max(1, int(round(max_attention * fraction))))
            attention_values = sorted(x for x in values if x <= max_attention)
        for attention_workers in attention_values:
            yield attention_workers, expert_workers


def _afd_stages(
    spec: DeploymentSpecification,
    attention: HardwareOption,
    expert: HardwareOption,
    attention_workers: int,
    expert_workers: int,
    batch_per_card: int,
    depth: int,
) -> Tuple[Dict[str, float], int, str]:
    model = spec.model
    microbatch = math.ceil(batch_per_card / depth)
    path = _communication_path(
        attention_workers + expert_workers,
        attention.hardware,
        expert.hardware,
    )
    bandwidth = _communication_bandwidth(path, attention.hardware, expert.hardware)
    payload = dispatch_or_combine_payloads(
        model,
        attention_workers,
        expert_workers,
        microbatch,
        path,
    )[2]
    routed = math.ceil(model.top_k * attention_workers * microbatch / expert_workers)
    communication = (
        payload
        / bandwidth
        * model.layers
        * spec.constraints.communication_scale
    )
    stages = {
        "attention": attention_layer_time(
            model, attention.hardware, microbatch, spec.seq_len
        )
        * model.layers,
        "dispatch": communication,
        "moe": expert_layer_time(model, expert.hardware, routed, expert_workers)
        * model.layers
        * spec.constraints.routing_tail_factor,
        "combine": communication,
    }
    return stages, microbatch, path


def _deep_pipeline_depths(
    spec: DeploymentSpecification,
    attention: HardwareOption,
    expert: HardwareOption,
    attention_workers: int,
    expert_workers: int,
    batch_per_card: int,
) -> Tuple[int, ...]:
    """Return an exact, sublinear set of deep-pipeline depths.

    For a fixed per-card batch, operator shapes depend on ``ceil(batch / p)``.
    Stage times are constant while that quotient is unchanged, and
    ``max(stage_sum, p * max_stage)`` is nondecreasing inside the interval.
    Therefore only the smallest depth in every quotient interval can be
    optimal.  Enumerating those boundaries is exhaustive-equivalent and uses
    O(sqrt(batch)) probes.
    """

    p0 = max(4, math.ceil(batch_per_card / spec.constraints.max_microbatch))
    pmax = max(p0, batch_per_card)

    depths = []
    depth = p0
    while depth <= pmax:
        depths.append(depth)
        microbatch = math.ceil(batch_per_card / depth)
        if microbatch <= 1:
            break
        interval_end = (batch_per_card - 1) // (microbatch - 1)
        depth = max(depth + 1, interval_end + 1)
    return tuple(depths)


def pipeline_depths(
    spec: DeploymentSpecification,
    attention: HardwareOption,
    expert: HardwareOption,
    attention_workers: int,
    expert_workers: int,
    batch_per_card: int,
) -> Tuple[int, ...]:
    shallow = range(1, min(3, batch_per_card) + 1)
    deep = _deep_pipeline_depths(
        spec,
        attention,
        expert,
        attention_workers,
        expert_workers,
        batch_per_card,
    )
    return tuple(sorted(set(shallow) | set(deep)))


def _afd_plan(
    spec: DeploymentSpecification,
    attention: HardwareOption,
    expert: HardwareOption,
    attention_workers: int,
    expert_workers: int,
    batch_per_card: int,
    depth: int,
) -> Optional[DeploymentPlan]:
    instance_cost = attention_workers * attention.price + expert_workers * expert.price
    replicas = int(math.floor((spec.budget + 1e-12) / instance_cost))
    if replicas < 1:
        return None
    budget_used = replicas * instance_cost
    if expert_workers > spec.model.experts:
        return None
    attention_memory = spec.model.attn_weight_bytes + kv_cache_bytes(
        spec.model, batch_per_card, spec.seq_len
    )
    expert_memory = spec.model.moe_weight_bytes_total / expert_workers
    if attention_memory > attention.hardware.hbm_capacity:
        return None
    if expert_memory > expert.hardware.hbm_capacity:
        return None

    stages, microbatch, path = _afd_stages(
        spec,
        attention,
        expert,
        attention_workers,
        expert_workers,
        batch_per_card,
        depth,
    )
    if microbatch > spec.constraints.max_microbatch:
        return None
    cycle, organization = afd_iteration_schedule(stages, depth)
    concurrency = replicas * attention_workers * batch_per_card
    return DeploymentPlan(
        architecture="afd",
        attention_hardware=attention.name,
        expert_hardware=expert.name,
        workers_attention=attention_workers,
        workers_expert=expert_workers,
        replicas=replicas,
        batch_per_attention_card=batch_per_card,
        microbatch_size=microbatch,
        pipeline_depth=depth,
        tpot_ms=cycle * 1e3,
        throughput_tps=concurrency / cycle,
        concurrency=concurrency,
        budget_used=budget_used,
        stage_attention_ms=stages["attention"] * 1e3,
        stage_dispatch_ms=stages["dispatch"] * 1e3,
        stage_expert_ms=stages["moe"] * 1e3,
        stage_combine_ms=stages["combine"] * 1e3,
        communication_path=path,
        execution_organization=organization,
    )


def enumerate_afd(
    spec: DeploymentSpecification,
    assignments: Optional[Iterable[Tuple[str, str]]] = None,
) -> Iterator[DeploymentPlan]:
    """Enumerate SLO-feasible AFD plans and optimize pipeline depth per shape."""

    model = spec.model
    if assignments is None:
        assignments = (
            (attention, expert)
            for attention in spec.hardware_menu
            for expert in spec.hardware_menu
        )
    for attention_name, expert_name in assignments:
        attention = spec.hardware_menu[attention_name]
        expert = spec.hardware_menu[expert_name]
        if not attention.supports_attention or not expert.supports_expert:
            continue
        kv_room = attention.hardware.hbm_capacity - model.attn_weight_bytes
        if kv_room <= 0:
            continue
        max_batch = int(kv_room // kv_cache_bytes(model, 1, spec.seq_len))
        if max_batch < 1:
            continue
        if spec.constraints.exhaustive_batches:
            batch_values = set(range(1, max_batch + 1))
        else:
            batch_values = set(value for value in spec.constraints.batch_candidates if value <= max_batch)
        batch_values.add(max_batch)
        for probe_depth in (1, 2, 3, 4, 6, 8, 12):
            batch_values.update(_batch_values(max_batch, spec.constraints, probe_depth))
        for attention_workers, expert_workers in _allocation_values(spec, attention, expert):
            for batch in sorted(batch_values):
                best_for_shape: Optional[DeploymentPlan] = None
                for depth in pipeline_depths(
                    spec,
                    attention,
                    expert,
                    attention_workers,
                    expert_workers,
                    batch,
                ):
                    plan = _afd_plan(
                        spec,
                        attention,
                        expert,
                        attention_workers,
                        expert_workers,
                        batch,
                        depth,
                    )
                    if plan is None:
                        continue
                    if best_for_shape is None or plan.tpot_ms < best_for_shape.tpot_ms:
                        best_for_shape = plan
                if best_for_shape is not None and best_for_shape.tpot_ms <= spec.tpot_slo_ms:
                    yield best_for_shape


def evaluate_afd_shape(
    spec: DeploymentSpecification,
    attention_hardware: str,
    expert_hardware: str,
    attention_workers: int,
    expert_workers: int,
    batch_per_card: int,
    pipeline_depth: int,
) -> DeploymentPlan:
    """Evaluate one explicit AFD organization for diagnostics and traces."""

    plan = _afd_plan(
        spec,
        spec.hardware_menu[attention_hardware],
        spec.hardware_menu[expert_hardware],
        attention_workers,
        expert_workers,
        batch_per_card,
        pipeline_depth,
    )
    if plan is None:
        raise ValueError("explicit AFD shape is infeasible")
    return plan


def pareto_prune(plans: Iterable[DeploymentPlan]) -> Tuple[DeploymentPlan, ...]:
    ordered = sorted(plans, key=lambda plan: (plan.tpot_ms, -plan.throughput_tps))
    frontier = []
    best_throughput = -1.0
    for plan in ordered:
        if plan.throughput_tps > best_throughput:
            frontier.append(plan)
            best_throughput = plan.throughput_tps
    return tuple(frontier)
