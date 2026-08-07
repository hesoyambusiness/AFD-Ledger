"""Precision helpers."""

from __future__ import annotations

from collections.abc import Collection


def dtype_bytes(dtype: str) -> int:
    key = dtype.lower()
    if key in {"fp4", "nvfp4", "mxfp4"}:
        return 1  # packed nibbles need layout detail; keep one byte conservative.
    if key in {"fp8", "int8", "uint8"}:
        return 1
    if key in {"fp16", "float16", "bf16", "bfloat16"}:
        return 2
    if key in {"fp32", "float32"}:
        return 4
    raise ValueError(f"unsupported dtype for byte size: {dtype}")


def normalize_precision(name: str) -> str:
    key = name.lower()
    if key in {"bfloat16", "bf16"}:
        return "bf16"
    if key in {"float16", "fp16"}:
        return "fp16"
    if key in {"float32", "fp32"}:
        return "fp32"
    if key in {"fp8", "int8", "fp4", "nvfp4", "mxfp4"}:
        return key
    return key


def resolve_compute_precision(requested: str, available: Collection[str]) -> str:
    """Resolve an arithmetic format without inventing a hardware capability."""

    precision = normalize_precision(requested)
    if precision in available:
        return precision
    if precision == "fp8" and "int8" in available:
        return "int8"
    choices = ", ".join(sorted(available))
    raise ValueError(
        f"requested compute precision {precision} is unavailable; available: {choices}"
    )
