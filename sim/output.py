"""CSV, summary, and plot output helpers."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

from .specs import Candidate, Hardware, ModelSpec


def generation_speed(c: Candidate) -> float:
    return c.generation_speed_tps_per_user


SeqLen = Union[int, float]


def candidate_row(c: Candidate, seq_len: SeqLen, model: ModelSpec, hw: Hardware) -> Dict[str, object]:
    row: Dict[str, object] = {
        "mode": c.mode,
        "tpot_ms": f"{c.tpot_ms:.6f}",
        "generation_speed_tps_per_user": f"{c.generation_speed_tps_per_user:.6f}",
        "throughput_tps": f"{c.throughput_tps:.6f}",
        "cards_used": c.cards_used,
        "replicas": c.replicas,
        "instance_cards": c.instance_cards,
        "m": c.m if c.m is not None else "",
        "n": c.n if c.n is not None else "",
        "collocated_ep_width": (
            c.collocated_ep_width if c.collocated_ep_width is not None else ""
        ),
        "microbatch_size": c.microbatch_size,
        "microbatch_num": c.microbatch_num,
        "per_card_batch": c.per_card_batch,
        "global_batch": c.global_batch,
        "memory_attn_gib": f"{c.memory_attn_gib:.6f}",
        "memory_expert_gib": f"{c.memory_expert_gib:.6f}",
        "memory_collocated_gib": f"{c.memory_collocated_gib:.6f}",
        "t_attn_ms": f"{c.t_attn_ms:.6f}",
        "t_moe_ms": f"{c.t_moe_ms:.6f}",
        "t_dispatch_ms": f"{c.t_dispatch_ms:.6f}",
        "t_combine_ms": f"{c.t_combine_ms:.6f}",
        "comm_path": c.comm_path,
        "overlap": c.overlap,
        "seq_len": seq_len,
        "model": model.name,
        "hardware": hw.name,
    }
    for key, value in c.extra.items():
        row[key] = f"{value:.6f}" if isinstance(value, float) else value
    return row


def write_candidates(path: Path, candidates: Sequence[Candidate], seq_len: SeqLen, model: ModelSpec, hw: Hardware) -> None:
    rows = [candidate_row(c, seq_len, model, hw) for c in candidates]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_by_sla(
    path: Path,
    rows: Sequence[Tuple[float, Optional[Candidate]]],
    seq_len: SeqLen,
    model: ModelSpec,
    hw: Hardware,
) -> None:
    converted = []
    for sla_ms, cand in rows:
        if cand is None:
            converted.append({"sla_ms": f"{sla_ms:.6f}", "feasible": "false"})
        else:
            row = candidate_row(cand, seq_len, model, hw)
            row["sla_ms"] = f"{sla_ms:.6f}"
            row["feasible"] = "true"
            converted.append(row)
    fieldnames = sorted({key for row in converted for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(converted)


def write_summary(
    path: Path,
    model: ModelSpec,
    hw: Hardware,
    seq_len: int,
    total_cards: int,
    candidates: Sequence[Candidate],
    afd_frontier: Sequence[Candidate],
    collocated_frontier: Sequence[Candidate],
) -> None:
    best_afd = max((c for c in candidates if c.mode == "afd"), key=lambda c: c.throughput_tps, default=None)
    best_collocated = max(
        (c for c in candidates if c.mode == "collocated"),
        key=lambda c: c.throughput_tps,
        default=None,
    )
    lines = [
        "# Simple Example Summary",
        "",
        f"model: {model.name}",
        f"hardware: {hw.name}",
        f"total_cards: {total_cards}",
        f"seq_len: {seq_len}",
        f"all_feasible_points: {len(candidates)}",
        f"afd_pareto_points: {len(afd_frontier)}",
        f"collocated_pareto_points: {len(collocated_frontier)}",
        "",
    ]
    if best_afd:
        lines.extend(
            [
                "best_afd_by_throughput:",
                f"  throughput_tps: {best_afd.throughput_tps:.3f}",
                f"  tpot_ms: {best_afd.tpot_ms:.3f}",
                f"  generation_speed_tps_per_user: {best_afd.generation_speed_tps_per_user:.3f}",
                f"  shape: replicas={best_afd.replicas}, m={best_afd.m}, n={best_afd.n}, "
                f"microbatch={best_afd.microbatch_size}x{best_afd.microbatch_num}",
                "",
            ]
        )
    if best_collocated:
        lines.extend(
            [
                "best_collocated_by_throughput:",
                f"  throughput_tps: {best_collocated.throughput_tps:.3f}",
                f"  tpot_ms: {best_collocated.tpot_ms:.3f}",
                f"  generation_speed_tps_per_user: {best_collocated.generation_speed_tps_per_user:.3f}",
                f"  shape: replicas={best_collocated.replicas}, "
                f"ep={best_collocated.collocated_ep_width}, "
                f"microbatch={best_collocated.microbatch_size}x{best_collocated.microbatch_num}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(
    output_dir: Path,
    afd_frontier: Sequence[Candidate],
    collocated_frontier: Sequence[Candidate],
    tbo_frontier: Sequence[Candidate],
    seq_len: int,
    dynamic_seq_var: float = 0.0,
    afd_dynamic_frontier: Sequence[Candidate] = (),
    collocated_dynamic_frontier: Sequence[Candidate] = (),
    tbo_dynamic_frontier: Sequence[Candidate] = (),
) -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def seq_label(value: float) -> str:
        return str(int(round(value))) if abs(value - round(value)) < 1e-6 else f"{value:.1f}"

    def plot_line(candidates: Sequence[Candidate], label: str, color: str, linestyle: str = "-") -> None:
        if candidates:
            ordered = sorted(candidates, key=generation_speed)
            ax.plot(
                [generation_speed(c) for c in ordered],
                [c.throughput_tps for c in ordered],
                marker="o",
                linewidth=1.6,
                markersize=3.5,
                linestyle=linestyle,
                color=color,
                label=label,
            )
            start = max(ordered, key=lambda c: (c.throughput_tps, generation_speed(c)))
            end = max(ordered, key=lambda c: (generation_speed(c), c.throughput_tps))
            ax.scatter(
                [generation_speed(start)],
                [start.throughput_tps],
                marker="s",
                s=60,
                color=color,
                edgecolors="black",
                linewidths=0.7,
                zorder=4,
            )
            ax.scatter(
                [generation_speed(end)],
                [end.throughput_tps],
                marker="*",
                s=120,
                color=color,
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"AFD": "tab:blue", "collocated": "tab:orange", "TBO": "tab:green"}
    plot_line(afd_frontier, "AFD", colors["AFD"])
    plot_line(collocated_frontier, "collocated", colors["collocated"])
    plot_line(tbo_frontier, "TBO", colors["TBO"])
    if dynamic_seq_var:
        runtime_seq_len = seq_len * (1.0 + dynamic_seq_var)
        plot_line(afd_dynamic_frontier, f"AFD runtime S={seq_label(runtime_seq_len)}", colors["AFD"], "--")
        plot_line(
            collocated_dynamic_frontier,
            f"collocated runtime S={seq_label(runtime_seq_len)}",
            colors["collocated"],
            "--",
        )
        plot_line(tbo_dynamic_frontier, f"TBO runtime S={seq_label(runtime_seq_len)}", colors["TBO"], "--")
    ax.set_xlabel("Generation speed (tokens/s/user)")
    ax.set_ylabel("System throughput (tokens/s)")
    ax.set_title(f"Decode throughput frontier\nseq_len={seq_len}; square=max throughput, star=max speed")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
