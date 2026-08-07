"""Seed-free comparison-aware provisioning over a hardware catalog."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .provisioning import (
    DeploymentPlan,
    DeploymentSpecification,
    enumerate_afd,
    enumerate_collocated,
)


HardwareAssignment = Tuple[str, str]


@dataclass(frozen=True)
class MenuFrontier:
    """Reusable architecture frontiers for one ordered hardware menu."""

    attention_hardware: str
    expert_hardware: str
    specification: DeploymentSpecification
    catalog_collocated: Tuple[DeploymentPlan, ...]
    afd: Tuple[DeploymentPlan, ...]

    @property
    def assignment(self) -> HardwareAssignment:
        return self.attention_hardware, self.expert_hardware

@dataclass(frozen=True)
class CatalogProvisioningResult:
    """The globally best AFD and collocated plans over a catalog."""

    specification: DeploymentSpecification
    tpot_slo_ms: float
    best_afd: Optional[DeploymentPlan]
    best_collocated: DeploymentPlan
    afd_plans: Mapping[HardwareAssignment, DeploymentPlan]
    collocated_plans: Mapping[str, DeploymentPlan]

    @property
    def best_afd_assignment(self) -> Optional[HardwareAssignment]:
        if self.best_afd is None:
            return None
        return self.best_afd.attention_hardware, self.best_afd.expert_hardware

    @property
    def comparison_gain(self) -> float:
        if self.best_afd is None:
            return 0.0
        return self.best_afd.throughput_tps / self.best_collocated.throughput_tps

    @property
    def selected_architecture(self) -> str:
        if (
            self.best_afd is not None
            and self.best_afd.throughput_tps > self.best_collocated.throughput_tps
        ):
            return "afd"
        return "collocated"

    @property
    def selected_plan(self) -> DeploymentPlan:
        if self.best_afd is not None and self.selected_architecture == "afd":
            return self.best_afd
        return self.best_collocated


def ordered_assignments(
    hardware_names: Iterable[str],
    *,
    include_homogeneous: bool = True,
) -> Tuple[HardwareAssignment, ...]:
    """Return the complete deterministic ordered AFD assignment space.

    AFD separates execution roles, not necessarily hardware types. General
    devices therefore contribute homogeneous ``(h, h)`` assignments, while
    role compatibility is enforced by the caller for specialized entries.
    """

    names = tuple(dict.fromkeys(hardware_names))
    return tuple(
        (attention, expert)
        for attention in names
        for expert in names
        if include_homogeneous or attention != expert
    )


def build_menu_frontiers(
    specification: DeploymentSpecification,
    assignments: Optional[Iterable[HardwareAssignment]] = None,
) -> Tuple[MenuFrontier, ...]:
    """Search every requested menu once at the specification's maximum SLO.

    The returned feasible plans can be reused for every tighter SLO. This makes
    an SLO sweep a selection operation rather than a repeated deployment search.
    """

    if assignments is None:
        assignments = ordered_assignments(specification.hardware_menu)

    collocated_menu = {
        name: option
        for name, option in specification.hardware_menu.items()
        if option.supports_collocated
    }
    if not collocated_menu:
        global_collocated: Tuple[DeploymentPlan, ...] = ()
    else:
        collocated_specification = replace(
            specification,
            hardware_menu=collocated_menu,
        )
        global_collocated = tuple(enumerate_collocated(collocated_specification))

    frontiers = []
    for attention, expert in assignments:
        attention_option = specification.hardware_menu[attention]
        expert_option = specification.hardware_menu[expert]
        if not attention_option.supports_attention:
            continue
        if not expert_option.supports_expert:
            continue
        paired = specification.with_menu((attention, expert))
        afd = tuple(enumerate_afd(paired, ((attention, expert),)))
        frontiers.append(
            MenuFrontier(
                attention,
                expert,
                specification,
                global_collocated,
                afd,
            )
        )
    return tuple(frontiers)


def _best_at_slo(
    plans: Sequence[DeploymentPlan],
    tpot_slo_ms: float,
) -> Optional[DeploymentPlan]:
    return max(
        (plan for plan in plans if plan.tpot_ms <= tpot_slo_ms),
        key=lambda plan: (plan.throughput_tps, -plan.tpot_ms),
        default=None,
    )


def provision_catalog_at_slo(
    frontiers: Sequence[MenuFrontier],
    tpot_slo_ms: float,
) -> CatalogProvisioningResult:
    """Choose the highest-throughput architecture and hardware organization."""

    if not frontiers:
        raise ValueError("catalog has no role-compatible ordered assignments")
    afd_plans = {}
    collocated_plans = {}
    for frontier in frontiers:
        afd = _best_at_slo(frontier.afd, tpot_slo_ms)
        if afd is not None:
            afd_plans[frontier.assignment] = afd
        for plan in frontier.catalog_collocated:
            if plan.tpot_ms > tpot_slo_ms:
                continue
            hardware = plan.attention_hardware
            current = collocated_plans.get(hardware)
            if current is None or (plan.throughput_tps, -plan.tpot_ms) > (
                current.throughput_tps,
                -current.tpot_ms,
            ):
                collocated_plans[hardware] = plan
    if not collocated_plans:
        raise ValueError(f"no collocated hardware is feasible at {tpot_slo_ms} ms")
    best_afd = max(
        afd_plans.values(),
        key=lambda plan: (plan.throughput_tps, -plan.tpot_ms),
        default=None,
    )
    best_collocated = max(
        collocated_plans.values(),
        key=lambda plan: (plan.throughput_tps, -plan.tpot_ms),
    )
    return CatalogProvisioningResult(
        replace(frontiers[0].specification, tpot_slo_ms=tpot_slo_ms),
        tpot_slo_ms,
        best_afd,
        best_collocated,
        afd_plans,
        collocated_plans,
    )


def provision_catalog(
    specification: DeploymentSpecification,
    assignments: Optional[Iterable[HardwareAssignment]] = None,
) -> CatalogProvisioningResult:
    """Seed-free exact catalog provisioning for one target SLO."""

    frontiers = build_menu_frontiers(specification, assignments)
    return provision_catalog_at_slo(frontiers, specification.tpot_slo_ms)
