"""Bounded seed-free hardware-pair search for throughput provisioning."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .catalog_provisioning import HardwareAssignment, ordered_assignments
from .operators import attention_layer_time, expert_layer_time, kv_cache_bytes
from .provisioning import (
    DeploymentPlan,
    DeploymentSpecification,
    HardwareOption,
    enumerate_afd,
    enumerate_collocated,
)


@dataclass(frozen=True)
class RoleProfile:
    hardware: str
    price: float
    service_rate: float
    service_per_price: float
    capacity_per_price: float


@dataclass(frozen=True)
class GuidedCatalogResult:
    specification: DeploymentSpecification
    best_afd: Optional[DeploymentPlan]
    best_collocated: DeploymentPlan
    evaluated_plans: Mapping[HardwareAssignment, DeploymentPlan]
    evaluation_order: Tuple[HardwareAssignment, ...]
    valid_assignments: int
    evaluation_budget: int

    @property
    def comparison_gain(self) -> float:
        if self.best_afd is None:
            return 0.0
        return self.best_afd.throughput_tps / self.best_collocated.throughput_tps

    @property
    def selected_architecture(self) -> str:
        if self.comparison_gain > 1.0:
            return "afd"
        return "collocated"

    @property
    def selected_plan(self) -> DeploymentPlan:
        if self.best_afd is not None and self.selected_architecture == "afd":
            return self.best_afd
        return self.best_collocated

    @property
    def evaluated_assignments(self) -> int:
        return len(self.evaluation_order)

    @property
    def feasible_evaluated_assignments(self) -> int:
        return len(self.evaluated_plans)


@dataclass(frozen=True)
class ProvisionRoundResult:
    """Feasible pair-best plans and all assignments charged to one round."""

    feasible_plans: Mapping[HardwareAssignment, DeploymentPlan]
    attempted_assignments: Tuple[HardwareAssignment, ...]


def _best(plans: Iterable[DeploymentPlan], architecture: str) -> DeploymentPlan:
    plan = max(
        plans,
        key=lambda candidate: (candidate.throughput_tps, -candidate.tpot_ms),
        default=None,
    )
    if plan is None:
        raise ValueError(f"no feasible {architecture} deployment")
    return plan


def _attention_profile(
    specification: DeploymentSpecification,
    option: HardwareOption,
) -> Optional[RoleProfile]:
    if not option.supports_attention:
        return None
    model = specification.model
    rates = []
    for batch in (1, 4, 16, 64):
        memory = model.attn_weight_bytes + kv_cache_bytes(
            model, batch, specification.seq_len
        )
        if memory > option.hardware.hbm_capacity:
            continue
        cycle = (
            attention_layer_time(
                model,
                option.hardware,
                batch,
                specification.seq_len,
            )
            * model.layers
        )
        rates.append(batch / cycle)
    if not rates:
        return None
    service = max(rates)
    return RoleProfile(
        option.name,
        option.price,
        service,
        service / option.price,
        option.hardware.hbm_capacity / option.price,
    )


def _expert_profile(
    specification: DeploymentSpecification,
    option: HardwareOption,
) -> Optional[RoleProfile]:
    if not option.supports_expert:
        return None
    model = specification.model
    minimum_workers = math.ceil(
        model.moe_weight_bytes_total / option.hardware.hbm_capacity
    )
    if minimum_workers > model.experts:
        return None
    ep_width = max(1, minimum_workers)
    rates = []
    for batch in (1, 4, 16, 64):
        cycle = (
            expert_layer_time(
                model,
                option.hardware,
                model.top_k * batch,
                ep_width,
            )
            * model.layers
        )
        rates.append(batch / cycle)
    service = max(rates)
    return RoleProfile(
        option.name,
        option.price,
        service,
        service / option.price,
        option.hardware.hbm_capacity / option.price,
    )


def _role_anchors(
    profiles: Mapping[str, RoleProfile],
) -> Tuple[str, ...]:
    if not profiles:
        return ()
    values = tuple(profiles.values())
    selected = {
        min(values, key=lambda profile: profile.price).hardware,
        max(values, key=lambda profile: profile.service_rate).hardware,
        max(values, key=lambda profile: profile.service_per_price).hardware,
        max(values, key=lambda profile: profile.capacity_per_price).hardware,
    }
    return tuple(sorted(selected))


def _pair_proxy(
    assignment: HardwareAssignment,
    attention_profiles: Mapping[str, RoleProfile],
    expert_profiles: Mapping[str, RoleProfile],
) -> float:
    attention = attention_profiles[assignment[0]].service_per_price
    expert = expert_profiles[assignment[1]].service_per_price
    return 2.0 / (1.0 / attention + 1.0 / expert)


def _valid_assignments(
    specification: DeploymentSpecification,
    attention_profiles: Mapping[str, RoleProfile],
    expert_profiles: Mapping[str, RoleProfile],
) -> Tuple[HardwareAssignment, ...]:
    return tuple(
        assignment
        for assignment in ordered_assignments(specification.hardware_menu)
        if assignment[0] in attention_profiles
        and assignment[1] in expert_profiles
    )


def _role_diverse_beam(
    evaluated: Mapping[HardwareAssignment, DeploymentPlan],
    beam_width: int,
) -> Tuple[HardwareAssignment, ...]:
    """Retain high-throughput pairs without collapsing onto one role basin."""

    ordered = sorted(
        evaluated,
        key=lambda assignment: evaluated[assignment].throughput_tps,
        reverse=True,
    )
    beam = []
    for assignment in ordered:
        if all(
            assignment[0] != retained[0] and assignment[1] != retained[1]
            for retained in beam
        ):
            beam.append(assignment)
            if len(beam) == beam_width:
                return tuple(beam)
    for assignment in ordered:
        if assignment not in beam:
            beam.append(assignment)
            if len(beam) == beam_width:
                break
    return tuple(beam)


def provision_round(
    specification: DeploymentSpecification,
    ranked_assignments: Iterable[HardwareAssignment],
    remaining_budget: int,
    attempted_assignments: Iterable[HardwareAssignment] = (),
) -> ProvisionRoundResult:
    """Fully provision a budgeted assignment list as Algorithm 2 requires."""

    if remaining_budget < 0:
        raise ValueError("remaining_budget must be non-negative")
    already_attempted = set(attempted_assignments)
    attempted = []
    feasible = {}
    for assignment in ranked_assignments:
        if len(attempted) == remaining_budget:
            break
        if assignment in already_attempted or assignment in attempted:
            continue
        attention, expert = assignment
        attempted.append(assignment)
        attention_option = specification.hardware_menu[attention]
        expert_option = specification.hardware_menu[expert]
        if not attention_option.supports_attention or not expert_option.supports_expert:
            continue
        paired = specification.with_menu(assignment)
        plans = tuple(enumerate_afd(paired, (assignment,)))
        if plans:
            feasible[assignment] = _best(plans, "AFD")
    return ProvisionRoundResult(dict(feasible), tuple(attempted))


def guided_provision_catalog(
    specification: DeploymentSpecification,
    *,
    evaluation_budget: int = 20,
    beam_width: int = 2,
    expansion_width: int = 6,
    max_rounds: int = 3,
) -> GuidedCatalogResult:
    """Search a bounded subset of ordered hardware pairs without a seed."""

    if evaluation_budget < 1:
        raise ValueError("evaluation_budget must be positive")
    if beam_width < 1 or expansion_width < 1 or max_rounds < 0:
        raise ValueError("beam and expansion parameters must be positive")

    collocated_menu = {
        name: option
        for name, option in specification.hardware_menu.items()
        if option.supports_collocated
    }
    if not collocated_menu:
        raise ValueError("catalog has no collocated-compatible hardware")
    collocated_specification = replace(
        specification,
        hardware_menu=collocated_menu,
    )
    best_collocated = _best(
        enumerate_collocated(collocated_specification),
        "collocated",
    )

    attention_profiles = {
        name: profile
        for name, option in specification.hardware_menu.items()
        if (profile := _attention_profile(specification, option)) is not None
    }
    expert_profiles = {
        name: profile
        for name, option in specification.hardware_menu.items()
        if (profile := _expert_profile(specification, option)) is not None
    }
    valid = _valid_assignments(
        specification,
        attention_profiles,
        expert_profiles,
    )
    if not valid:
        return GuidedCatalogResult(
            specification,
            None,
            best_collocated,
            {},
            (),
            0,
            0,
        )
    valid_set = set(valid)
    budget = min(evaluation_budget, len(valid))

    attention_anchors = _role_anchors(attention_profiles)
    expert_anchors = _role_anchors(expert_profiles)
    anchors = {
        (attention, expert)
        for attention in attention_anchors
        for expert in expert_anchors
        if (attention, expert) in valid_set
    }
    proxy_order = sorted(
        valid,
        key=lambda assignment: (
            _pair_proxy(assignment, attention_profiles, expert_profiles),
            assignment,
        ),
        reverse=True,
    )

    evaluated: Dict[HardwareAssignment, DeploymentPlan] = {}
    evaluation_order = []

    def evaluate_ranked(assignments: Iterable[HardwareAssignment]) -> None:
        round_result = provision_round(
            specification,
            assignments,
            budget - len(evaluation_order),
            evaluation_order,
        )
        evaluation_order.extend(round_result.attempted_assignments)
        evaluated.update(round_result.feasible_plans)

    starting_pairs = sorted(
        anchors,
        key=lambda candidate: (
            _pair_proxy(candidate, attention_profiles, expert_profiles),
            candidate,
        ),
        reverse=True,
    )
    evaluate_ranked(starting_pairs)

    for _ in range(max_rounds):
        if len(evaluation_order) >= budget or not evaluated:
            break
        beam = _role_diverse_beam(evaluated, beam_width)
        neighbors = {
            candidate
            for candidate in valid
            if candidate not in evaluation_order
            and any(
                candidate[0] == current[0] or candidate[1] == current[1]
                for current in beam
            )
        }
        ranked = sorted(
            neighbors,
            key=lambda assignment: (
                _pair_proxy(assignment, attention_profiles, expert_profiles),
                assignment,
            ),
            reverse=True,
        )
        evaluate_ranked(ranked[:expansion_width])

    evaluate_ranked(proxy_order)

    best_afd = _best(evaluated.values(), "AFD") if evaluated else None
    return GuidedCatalogResult(
        specification,
        best_afd,
        best_collocated,
        dict(evaluated),
        tuple(evaluation_order),
        len(valid),
        budget,
    )
