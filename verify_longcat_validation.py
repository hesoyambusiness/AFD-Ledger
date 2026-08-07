#!/usr/bin/env python3
"""Regenerate the LongCat analytical-versus-production validation table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sim.longcat_validation import validation_rows


def verify(output_dir: Path) -> list[dict]:
    rows = validation_rows()
    if len(rows) != 3:
        raise AssertionError(f"expected three 160-card cases, found {len(rows)}")
    for row in rows:
        modeled = float(row["theta_model_ratio"])
        measured = float(row["theta_measured_ratio"])
        if (modeled > 1.0) != (measured > 1.0):
            raise AssertionError("model and measurement disagree on architecture")
        relative = modeled / measured
        if not 1.06 <= relative <= 1.10:
            raise AssertionError(f"unexpected model bias: {relative:.6f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "longcat_validation.csv"
    fields = sorted({field for row in rows for field in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/longcat_validation"),
    )
    args = parser.parse_args()
    rows = verify(args.output_dir)
    for row in rows:
        print(
            f"{row['input_len']}: measured={float(row['theta_measured_ratio']):.3f} "
            f"modeled={float(row['theta_model_ratio']):.3f} "
            f"bias={float(row['model_over_measured_ratio']):.3f}"
        )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
