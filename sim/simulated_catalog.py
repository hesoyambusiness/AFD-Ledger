"""Build the fixed role-specialized catalog used by provisioning studies."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .provisioning import HardwareOption


def _specialized_option(
    name: str,
    specification: Mapping[str, object],
    base: HardwareOption,
) -> HardwareOption:
    role = str(specification["role"])
    if role not in {"attention", "expert"}:
        raise ValueError(f"simulated hardware {name}: invalid role {role!r}")
    network_bw = float(specification["network_bytes_per_s"])
    fp8_flops = float(specification["fp8_flops"])
    hardware = replace(
        base.hardware,
        name=name,
        hbm_capacity=float(specification["memory_bytes"]),
        hbm_bw=float(specification["hbm_bytes_per_s"]),
        peak_flops={"bf16": fp8_flops / 2.0, "fp8": fp8_flops},
        intra_node_bw=max(network_bw, base.hardware.intra_node_bw),
        inter_node_bw=network_bw,
        inter_rack_bw=network_bw,
        gpus_per_node=8,
    )
    return HardwareOption(
        name,
        hardware,
        float(specification["price_h200_normalized"]),
        frozenset({"simulated", f"{role}_only"}),
    )


def fixed_simulated_catalog(
    general_catalog: Mapping[str, HardwareOption],
    configuration: Mapping[str, object],
) -> Mapping[str, HardwareOption]:
    """Return one immutable-input catalog with all simulated entries present."""

    general_names = tuple(configuration["general_devices"])
    missing = set(general_names) - set(general_catalog)
    if missing:
        raise ValueError(
            f"simulated catalog is missing general devices: {', '.join(sorted(missing))}"
        )
    catalog = {name: general_catalog[name] for name in general_names}
    for name, device in configuration["devices"].items():
        role = str(device["role"])
        base_name = "b200_sxm" if role == "attention" else "l40s"
        catalog[name] = _specialized_option(name, device, catalog[base_name])
    return catalog
