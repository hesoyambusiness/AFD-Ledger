"""Load the consistently priced commercial catalog for throughput studies."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

from .config import load_hardware, resolve_hardware_config_path
from .provisioning import HardwareOption


def load_priced_existing_catalog(
    configuration: Mapping[str, object],
    names: Optional[Sequence[str]] = None,
) -> Mapping[str, HardwareOption]:
    config = configuration["existing_catalog"]
    prices = config["prices_usd_per_hour"]
    normalizer = str(config["price_normalizer"])
    normalizer_price = float(prices[normalizer])
    selected = tuple(prices) if names is None else tuple(names)
    return {
        name: HardwareOption(
            name,
            load_hardware(resolve_hardware_config_path(Path(name))),
            float(prices[name]) / normalizer_price,
        )
        for name in selected
    }
