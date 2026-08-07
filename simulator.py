#!/usr/bin/env python3
"""Decode-only AFD versus collocated MoE Pareto simulator.

The simulator intentionally follows the simplified model documented in this
directory:

* decode only; prefill is out of scope
* P/D disaggregation is assumed
* operator latency follows strict roofline
* AFD memory: attention weights + KV cache on attention cards, MoE weights on
  expert cards
* collocated memory: attention weights + MoE shard + KV cache on every card
* both modes use wide EP, so one MoE instance cannot exceed the expert count
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence

from sim.config import load_hardware, load_model, resolve_hardware_config_path
from sim.output import plot_results, write_best_by_sla, write_candidates, write_summary
from sim.pareto import best_by_sla, dynamic_sequence_candidates, dynamic_sequence_frontier, pareto_frontier
from sim.search import enumerate_afd, enumerate_collocated

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.mplconfig"))


def default_model_path() -> Path:
    return ROOT / "configs/models/qwen3_30b_a3b_fp8.json"


def default_hardware_path() -> Path:
    return ROOT / "configs/hardware/h200_sxm.json"


def parse_sla_grid(value: str) -> List[float]:
    if ":" in value:
        start, stop, count = value.split(":")
        start_f = float(start)
        stop_f = float(stop)
        count_i = int(count)
        if count_i <= 1:
            return [start_f]
        step = (stop_f - start_f) / (count_i - 1)
        return [start_f + i * step for i in range(count_i)]
    return [float(part) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=default_model_path(),
        help=(
            "model config path or local configs/models model id/name, e.g. "
            "Qwen/Qwen3-235B-A22B or deepseek-ai/DeepSeek-V3.2"
        ),
    )
    parser.add_argument(
        "--hardware-config",
        type=Path,
        default=default_hardware_path(),
        help="hardware JSON config path or local configs/hardware name",
    )
    parser.add_argument("--total-cards", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=40960, help="single decode sequence length used for attention latency and KV memory")
    parser.add_argument("--max-microbatch-size", type=int, default=64)
    parser.add_argument("--full-search", action="store_true", help="search every integer split and microbatch size")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--compute-precision", default=None)
    parser.add_argument("--attention-precision", default="bf16")
    parser.add_argument("--activation-precision", default="bf16")
    parser.add_argument("--kv-precision", default="bf16")
    parser.add_argument(
        "--seq-len-alpha",
        type=float,
        default=0.0,
        help="compatibility alias for --dynamic-seq-var",
    )
    parser.add_argument(
        "--dynamic-seq-var",
        type=float,
        default=None,
        help=(
            "runtime-only sequence-length variance. Search still uses --seq-len; "
            "the dynamic view recomputes only per-microbatch attention latency at "
            "(1 + var) * seq_len."
        ),
    )
    parser.add_argument("--sla-grid-ms", default="5:200:40")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/simple_example")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.seq_len_alpha < 0:
        parser.error("--seq-len-alpha must be non-negative")
    dynamic_seq_var = args.seq_len_alpha if args.dynamic_seq_var is None else args.dynamic_seq_var
    if dynamic_seq_var < 0:
        parser.error("--dynamic-seq-var must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hw = load_hardware(resolve_hardware_config_path(args.hardware_config))
    model = load_model(
        args.model_config,
        compute_precision=args.compute_precision,
        attention_precision=args.attention_precision,
        activation_precision=args.activation_precision,
        kv_precision=args.kv_precision,
    )
    seq_len = args.seq_len
    kv_seq_len = seq_len
    runtime_seq_len = seq_len * (1.0 + dynamic_seq_var)
    sla_grid = parse_sla_grid(args.sla_grid_ms)

    afd = enumerate_afd(
        model=model,
        hw=hw,
        seq_len=seq_len,
        kv_seq_len=kv_seq_len,
        total_cards=args.total_cards,
        max_microbatch_size=args.max_microbatch_size,
        beta=args.beta,
        compact=not args.full_search,
        sla_s=[tau_ms * 1e-3 for tau_ms in sla_grid],
    )
    collocated = enumerate_collocated(
        model=model,
        hw=hw,
        seq_len=seq_len,
        kv_seq_len=kv_seq_len,
        total_cards=args.total_cards,
        max_microbatch_size=args.max_microbatch_size,
        beta=args.beta,
        compact=not args.full_search,
    )
    all_candidates = sorted(
        afd + collocated,
        key=lambda c: (c.mode, c.tpot_s, -c.throughput_tps),
    )
    afd_frontier = pareto_frontier(afd)
    collocated_sequential_frontier = pareto_frontier(
        [c for c in collocated if c.microbatch_num == 1]
    )
    collocated_tbo_frontier = pareto_frontier(
        [c for c in collocated if c.microbatch_num == 2]
    )
    afd_dynamic = dynamic_sequence_candidates(afd, runtime_seq_len, seq_len, model, hw) if dynamic_seq_var else []
    collocated_dynamic = (
        dynamic_sequence_candidates(
            collocated, runtime_seq_len, seq_len, model, hw
        )
        if dynamic_seq_var
        else []
    )
    afd_dynamic_frontier = (
        dynamic_sequence_frontier(afd, runtime_seq_len, seq_len, model, hw)
        if dynamic_seq_var
        else []
    )
    collocated_sequential_dynamic_frontier = (
        dynamic_sequence_frontier(
            [c for c in collocated if c.microbatch_num == 1],
            runtime_seq_len,
            seq_len,
            model,
            hw,
        )
        if dynamic_seq_var
        else []
    )
    collocated_tbo_dynamic_frontier = (
        dynamic_sequence_frontier(
            [c for c in collocated if c.microbatch_num == 2],
            runtime_seq_len,
            seq_len,
            model,
            hw,
        )
        if dynamic_seq_var
        else []
    )
    collocated_frontier = pareto_frontier(collocated)
    combined_frontier = pareto_frontier(all_candidates)
    dynamic_combined_frontier = (
        pareto_frontier(afd_dynamic + collocated_dynamic)
        if dynamic_seq_var
        else []
    )
    best_sla = best_by_sla(all_candidates, sla_grid)
    dynamic_best_sla = (
        best_by_sla(afd_dynamic + collocated_dynamic, sla_grid)
        if dynamic_seq_var
        else []
    )

    write_candidates(args.output_dir / "all_feasible.csv", all_candidates, seq_len, model, hw)
    write_candidates(args.output_dir / "pareto_afd.csv", afd_frontier, seq_len, model, hw)
    write_candidates(
        args.output_dir / "pareto_collocated.csv",
        collocated_frontier,
        seq_len,
        model,
        hw,
    )
    write_candidates(args.output_dir / "pareto.csv", combined_frontier, seq_len, model, hw)
    write_best_by_sla(args.output_dir / "best_by_sla.csv", best_sla, seq_len, model, hw)
    if dynamic_seq_var:
        write_candidates(
            args.output_dir / "dynamic_all_feasible.csv",
            afd_dynamic + collocated_dynamic,
            runtime_seq_len,
            model,
            hw,
        )
        write_candidates(args.output_dir / "dynamic_pareto_afd.csv", afd_dynamic_frontier, runtime_seq_len, model, hw)
        write_candidates(
            args.output_dir / "dynamic_pareto_collocated.csv",
            pareto_frontier(collocated_dynamic),
            runtime_seq_len,
            model,
            hw,
        )
        write_candidates(args.output_dir / "dynamic_pareto.csv", dynamic_combined_frontier, runtime_seq_len, model, hw)
        write_best_by_sla(args.output_dir / "dynamic_best_by_sla.csv", dynamic_best_sla, runtime_seq_len, model, hw)
    write_summary(
        args.output_dir / "summary.md",
        model,
        hw,
        seq_len,
        args.total_cards,
        all_candidates,
        afd_frontier,
        collocated_frontier,
    )
    if not args.no_plot:
        plot_results(
            args.output_dir,
            afd_frontier,
            collocated_sequential_frontier,
            collocated_tbo_frontier,
            seq_len,
            dynamic_seq_var,
            afd_dynamic_frontier,
            collocated_sequential_dynamic_frontier,
            collocated_tbo_dynamic_frontier,
        )

    print(f"model={model.name}")
    print(f"hardware={hw.name}")
    print(f"seq_len={seq_len}")
    if dynamic_seq_var:
        print(f"kv_allocation_seq_len={kv_seq_len:.3f}")
        print(f"runtime_seq_len={runtime_seq_len:.3f}")
    print(
        f"feasible: afd={len(afd)} collocated={len(collocated)} "
        f"all={len(all_candidates)}"
    )
    print(
        f"pareto: afd={len(afd_frontier)} "
        f"collocated={len(collocated_frontier)} "
        f"combined={len(combined_frontier)}"
    )
    if dynamic_seq_var:
        print(
            f"dynamic_seq: var={dynamic_seq_var} runtime_seq_len={runtime_seq_len:.3f} "
            f"afd={len(afd_dynamic_frontier)} "
            f"collocated={len(collocated_sequential_dynamic_frontier)} "
            f"tbo={len(collocated_tbo_dynamic_frontier)} "
            f"combined={len(dynamic_combined_frontier)}"
        )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
