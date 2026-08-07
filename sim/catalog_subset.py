"""Existing-hardware catalog subset selection for deployment scenarios."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence, Tuple

from .catalog_provisioning import CatalogProvisioningResult
from .provisioning import DeploymentPlan


@dataclass(frozen=True)
class ExistingSubsetResult:
    hardware: Tuple[str, ...]
    best_afd: DeploymentPlan
    best_collocated: DeploymentPlan

    @property
    def comparison_gain(self) -> float:
        return self.best_afd.throughput_tps / self.best_collocated.throughput_tps


def select_existing_subset(
    provisioned_catalog: CatalogProvisioningResult,
    hardware_names: Sequence[str],
    *,
    min_size: int = 2,
    max_size: int = 3,
    minimum_gain: float = 1.0,
) -> ExistingSubsetResult:
    """Find the largest availability subset meeting an AFD-gain requirement.

    The complete catalog is provisioned once. Subset selection then reuses the
    exact per-assignment and per-device winners because removing hardware does
    not change any retained deployment's feasibility or performance.
    """

    names = tuple(dict.fromkeys(hardware_names))
    max_size = min(max_size, len(names))
    candidates = []
    for size in range(min_size, max_size + 1):
        for subset in itertools.combinations(names, size):
            allowed = set(subset)
            afd_plans = [
                plan
                for assignment, plan in provisioned_catalog.afd_plans.items()
                if assignment[0] in allowed and assignment[1] in allowed
            ]
            collocated_plans = [
                plan
                for hardware, plan in provisioned_catalog.collocated_plans.items()
                if hardware in allowed
            ]
            if not afd_plans or not collocated_plans:
                continue
            best_afd = max(
                afd_plans,
                key=lambda plan: (plan.throughput_tps, -plan.tpot_ms),
            )
            best_collocated = max(
                collocated_plans,
                key=lambda plan: (plan.throughput_tps, -plan.tpot_ms),
            )
            result = ExistingSubsetResult(subset, best_afd, best_collocated)
            if result.comparison_gain >= minimum_gain:
                candidates.append(result)

    if not candidates:
        raise ValueError(
            "no existing-hardware subset satisfies the minimum global gain"
        )
    return max(
        candidates,
        key=lambda result: (
            len(result.hardware),
            result.comparison_gain,
            result.best_afd.throughput_tps,
        ),
    )
