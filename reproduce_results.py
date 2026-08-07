#!/usr/bin/env python3
"""Regenerate the public artifacts used by the paper's Evaluation section."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str, output_dir: Path, *arguments: str) -> None:
    command = [
        sys.executable,
        str(ROOT / script),
        "--output-dir",
        str(output_dir),
        *arguments,
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/reproduction",
    )
    args = parser.parse_args()
    output = args.output_dir
    run("verify_longcat_validation.py", output / "longcat_validation")
    run(
        "verify_guided_search.py",
        output / "guided_search",
        "--evaluation-budget",
        "20",
    )
    run("verify_existing_subset.py", output / "existing_hardware_subset")
    run("verify_homogeneous_afd.py", output / "homogeneous_afd")
    run("verify_simulated_catalog.py", output / "simulated_catalog")
    run("verify_catalog_provisioning.py", output / "simulated_slo_sweep")
    run(
        "verify_extended_results.py",
        output / "extended_results",
        "--only",
        "evaluation",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
