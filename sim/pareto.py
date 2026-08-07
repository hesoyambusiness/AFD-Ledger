"""Pareto frontier utilities."""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

from .operators import runtime_attention_layer_time
from .schedules import afd_iteration_schedule
from .specs import Candidate, Hardware, ModelSpec


def pareto_frontier(candidates: Sequence[Candidate]) -> List[Candidate]:
    ordered = sorted(candidates, key=lambda c: (c.tpot_s, -c.throughput_tps))
    frontier = []
    best_throughput = -1.0
    for cand in ordered:
        if cand.throughput_tps > best_throughput:
            frontier.append(cand)
            best_throughput = cand.throughput_tps
    return frontier


def best_by_sla(candidates: Sequence[Candidate], sla_ms: Sequence[float]) -> List[Tuple[float, Optional[Candidate]]]:
    result = []
    for tau_ms in sla_ms:
        feasible = [c for c in candidates if c.tpot_ms <= tau_ms]
        result.append((tau_ms, max(feasible, key=lambda c: c.throughput_tps) if feasible else None))
    return result


def with_runtime_sequence(
    candidate: Candidate,
    runtime_seq_len: float,
    nominal_seq_len: int,
    model: ModelSpec,
    hw: Hardware,
) -> Candidate:
    t_attn_s = runtime_attention_layer_time(
        model,
        hw,
        candidate.microbatch_size,
        nominal_seq_len,
        runtime_seq_len,
    ) * model.layers
    t_moe_s = candidate.t_moe_ms * 1e-3
    t_dispatch_s = candidate.t_dispatch_ms * 1e-3
    t_combine_s = candidate.t_combine_ms * 1e-3
    extra = dict(candidate.extra)
    dynamic_seq_var = runtime_seq_len / nominal_seq_len - 1.0 if nominal_seq_len > 0 else 0.0

    if candidate.mode == "afd":
        stage_times = {
            "attention": t_attn_s,
            "dispatch": t_dispatch_s,
            "moe": t_moe_s,
            "combine": t_combine_s,
        }
        max_stage_time = max(stage_times.values())
        t_iter, overlap = afd_iteration_schedule(stage_times, candidate.microbatch_num)
        extra.update(
            {
                "dynamic_seq_var": dynamic_seq_var,
                "nominal_seq_len": nominal_seq_len,
                "runtime_seq_len": runtime_seq_len,
                "runtime_changes_attention_only": "true",
                "dsa_sparse_attention_uses_nominal_topk_window": str(model.attention_kind == "dsa").lower(),
                "afd_max_stage_ms": max_stage_time * 1e3,
            }
        )
    elif candidate.microbatch_num == 1:
        t_iter = t_attn_s + t_moe_s + t_dispatch_s + t_combine_s
        overlap = "none"
        extra.update(
            {
                "dynamic_seq_var": dynamic_seq_var,
                "nominal_seq_len": nominal_seq_len,
                "runtime_seq_len": runtime_seq_len,
                "runtime_changes_attention_only": "true",
                "dsa_sparse_attention_uses_nominal_topk_window": str(model.attention_kind == "dsa").lower(),
                "compute_full_ms": (t_attn_s + t_moe_s) * 1e3,
                "comm_full_ms": (t_dispatch_s + t_combine_s) * 1e3,
            }
        )
    else:
        tbo_stage_1 = max(t_attn_s, t_combine_s)
        tbo_stage_2 = max(t_dispatch_s, t_attn_s)
        tbo_stage_3 = max(t_moe_s, t_dispatch_s)
        tbo_stage_4 = max(t_combine_s, t_moe_s)
        t_iter = tbo_stage_1 + tbo_stage_2 + tbo_stage_3 + tbo_stage_4
        if t_combine_s <= t_attn_s and t_dispatch_s <= t_attn_s and t_dispatch_s <= t_moe_s and t_combine_s <= t_moe_s:
            overlap = "two_batch_fully_hidden"
        else:
            overlap = "two_batch_partially_hidden"
        extra.update(
            {
                "dynamic_seq_var": dynamic_seq_var,
                "nominal_seq_len": nominal_seq_len,
                "runtime_seq_len": runtime_seq_len,
                "runtime_changes_attention_only": "true",
                "dsa_sparse_attention_uses_nominal_topk_window": str(model.attention_kind == "dsa").lower(),
                "compute_full_ms": (t_attn_s + t_moe_s) * 1e3,
                "comm_full_ms": (t_dispatch_s + t_combine_s) * 1e3,
                "tbo_stage_1_max_attn_prev_combine_ms": tbo_stage_1 * 1e3,
                "tbo_stage_2_max_dispatch_attn_ms": tbo_stage_2 * 1e3,
                "tbo_stage_3_max_moe_dispatch_ms": tbo_stage_3 * 1e3,
                "tbo_stage_4_max_combine_moe_ms": tbo_stage_4 * 1e3,
            }
        )

    return replace(
        candidate,
        tpot_s=t_iter,
        throughput_tps=candidate.global_batch / t_iter,
        t_attn_ms=t_attn_s * 1e3,
        overlap=overlap,
        extra=extra,
    )


def dynamic_sequence_candidates(
    candidates: Sequence[Candidate],
    runtime_seq_len: float,
    nominal_seq_len: int,
    model: ModelSpec,
    hw: Hardware,
) -> List[Candidate]:
    return [with_runtime_sequence(c, runtime_seq_len, nominal_seq_len, model, hw) for c in candidates]


def dynamic_sequence_frontier(
    candidates: Sequence[Candidate],
    runtime_seq_len: float,
    nominal_seq_len: int,
    model: ModelSpec,
    hw: Hardware,
) -> List[Candidate]:
    return pareto_frontier(dynamic_sequence_candidates(candidates, runtime_seq_len, nominal_seq_len, model, hw))
