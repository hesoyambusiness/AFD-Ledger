"""Search-space enumeration for AFD and collocated decode modes."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .operators import (
    attention_layer_time,
    comm_path_for_cards,
    dispatch_or_combine_payloads,
    expert_layer_time,
    kv_cache_bytes,
    memory_gib,
    collocated_comm_time,
)
from .schedules import afd_iteration_schedule
from .specs import Candidate, Hardware, ModelSpec


def add_if_finite(candidates: List[Candidate], cand: Candidate) -> None:
    if math.isfinite(cand.tpot_s) and math.isfinite(cand.throughput_tps) and cand.throughput_tps > 0:
        candidates.append(cand)


def divisors_leq(n: int, limit: int) -> List[int]:
    return [x for x in range(1, limit + 1) if n % x == 0]


def candidate_batch_sizes(max_value: int, compact: bool) -> List[int]:
    if max_value < 1:
        return []
    if not compact:
        return list(range(1, max_value + 1))
    values = [1, 2, 4, 8, 16, 32, 64, 128]
    return [v for v in values if v <= max_value]


def candidate_microbatch_nums(max_value: int, compact: bool, afd: bool) -> List[int]:
    if not compact:
        return list(range(1, min(max_value, 3) + 1)) if afd else list(range(1, max_value + 1))
    if afd:
        return list(range(1, min(max_value, 3) + 1))
    return [1, 2]


def afd_network_coefficient(model: ModelSpec, hw: Hardware, m: int, n: int, path: str) -> float:
    return model.layers * model.top_k * model.hidden * model.act_bytes * min(1.0, m / n) / hw.comm_bw(path)


def afd_stage_times(
    model: ModelSpec,
    hw: Hardware,
    seq_len: int,
    m: int,
    n: int,
    b_mu: int,
    beta: float,
    path: str,
) -> tuple[dict[str, float], float, float, float]:
    b_mu_global = m * b_mu
    routed_per_expert_card = math.ceil(beta * model.top_k * b_mu_global / n)
    t_attn_layer = attention_layer_time(model, hw, b_mu, seq_len)
    t_expert_layer = expert_layer_time(model, hw, routed_per_expert_card, n)
    (
        attn_side_payload,
        expert_side_payload,
        selected_comm_payload,
    ) = dispatch_or_combine_payloads(model, m, n, b_mu, path)
    t_dispatch_layer = selected_comm_payload / hw.comm_bw(path)
    t_combine_layer = t_dispatch_layer
    return (
        {
            "attention": t_attn_layer * model.layers,
            "dispatch": t_dispatch_layer * model.layers,
            "moe": t_expert_layer * model.layers,
            "combine": t_combine_layer * model.layers,
        },
        attn_side_payload,
        expert_side_payload,
        selected_comm_payload,
    )


def make_afd_candidate(
    model: ModelSpec,
    hw: Hardware,
    seq_len: int,
    kv_seq_len: float,
    total_cards: int,
    cards: int,
    replicas: int,
    m: int,
    n: int,
    b_card: int,
    microbatch_num: int,
    beta: float,
    path: str,
    attn_weight: float,
    expert_mem: float,
    max_microbatch_size: int,
    search_policy: str,
) -> Optional[Candidate]:
    if microbatch_num < 1 or b_card < 1:
        return None
    b_mu = math.ceil(b_card / microbatch_num)
    if b_mu < 1 or b_mu > max_microbatch_size:
        return None

    stage_times, attn_side_payload, expert_side_payload, selected_comm_payload = afd_stage_times(
        model,
        hw,
        seq_len,
        m,
        n,
        b_mu,
        beta,
        path,
    )
    max_stage_time = max(stage_times.values())

    t_iter, overlap = afd_iteration_schedule(stage_times, microbatch_num)

    global_batch = m * b_card
    inst_throughput = global_batch / t_iter
    return Candidate(
        mode="afd",
        tpot_s=t_iter,
        throughput_tps=replicas * inst_throughput,
        cards_used=replicas * cards,
        replicas=replicas,
        instance_cards=cards,
        m=m,
        n=n,
        microbatch_size=b_mu,
        microbatch_num=microbatch_num,
        per_card_batch=b_card,
        global_batch=replicas * global_batch,
        memory_attn_gib=memory_gib(attn_weight + kv_cache_bytes(model, b_card, kv_seq_len)),
        memory_expert_gib=memory_gib(expert_mem),
        t_attn_ms=stage_times["attention"] * 1e3,
        t_moe_ms=stage_times["moe"] * 1e3,
        t_dispatch_ms=stage_times["dispatch"] * 1e3,
        t_combine_ms=stage_times["combine"] * 1e3,
        comm_path=path,
        overlap=overlap,
        extra={
            "afd_max_stage_ms": max_stage_time * 1e3,
            "afd_comm_attention_payload_bytes": attn_side_payload,
            "afd_comm_expert_payload_bytes": expert_side_payload,
            "afd_comm_selected_payload_bytes": selected_comm_payload,
            "afd_comm_payload_mode": "push_pull_min_integer_token",
            "afd_search_policy": search_policy,
            "nominal_attention_seq_len": seq_len,
            "kv_allocation_seq_len": kv_seq_len,
        },
    )


def enumerate_afd(
    model: ModelSpec,
    hw: Hardware,
    seq_len: int,
    kv_seq_len: float,
    total_cards: int,
    max_microbatch_size: int,
    beta: float,
    compact: bool,
    sla_s: Sequence[float] = (),
) -> List[Candidate]:
    candidates: List[Candidate] = []
    hbm = hw.hbm_capacity
    attn_weight = model.attn_weight_bytes
    moe_weight = model.moe_weight_bytes_total

    if compact:
        n_values = divisors_leq(model.experts, min(model.experts, total_cards - 1))
        instance_sizes = divisors_leq(total_cards, total_cards)
    else:
        n_values = list(range(1, min(model.experts, total_cards - 1) + 1))
        instance_sizes = list(range(2, total_cards + 1))

    for n in n_values:
        expert_mem = moe_weight / n
        if expert_mem > hbm:
            continue
        for cards in instance_sizes:
            if cards <= n:
                continue
            m = cards - n
            cards = m + n
            replicas = total_cards // cards
            if replicas < 1:
                continue
            path = comm_path_for_cards(cards, hw)
            kv_room = hbm - attn_weight
            if kv_room <= 0:
                continue
            max_b_card_by_mem = int(kv_room // kv_cache_bytes(model, 1, kv_seq_len))
            if max_b_card_by_mem < 1:
                continue
            for microbatch_num in candidate_microbatch_nums(3, compact, afd=True):
                max_b_mu = min(max_microbatch_size, max_b_card_by_mem // microbatch_num)
                if max_b_mu < 1:
                    continue
                for b_mu in candidate_batch_sizes(max_b_mu, compact):
                    cand = make_afd_candidate(
                        model,
                        hw,
                        seq_len,
                        kv_seq_len,
                        total_cards,
                        cards,
                        replicas,
                        m,
                        n,
                        b_mu * microbatch_num,
                        microbatch_num,
                        beta,
                        path,
                        attn_weight,
                        expert_mem,
                        max_microbatch_size,
                        f"small_p{microbatch_num}",
                    )
                    if cand is not None:
                        add_if_finite(candidates, cand)

            p0 = 4
            a_net = afd_network_coefficient(model, hw, m, n, path)
            tau_bounds = sorted({tau for tau in sla_s if tau > 0}, reverse=True)
            if not tau_bounds:
                tau_bounds = [float("inf")]
            seen_deep: dict[int, float] = {}
            for tau in tau_bounds:
                if math.isfinite(tau):
                    b_card_upper = min(max_b_card_by_mem, int(tau // a_net) if a_net > 0 else max_b_card_by_mem)
                else:
                    b_card_upper = max_b_card_by_mem
                for b_card in range(b_card_upper, 0, -1):
                    if b_card in seen_deep and (not math.isfinite(tau) or seen_deep[b_card] <= tau):
                        break
                    p_min = max(p0, math.ceil(b_card / max_microbatch_size))
                    p_max = max(p0, b_card)
                    best_cand = None
                    for microbatch_num in range(p_min, p_max + 1):
                        cand = make_afd_candidate(
                            model,
                            hw,
                            seq_len,
                            kv_seq_len,
                            total_cards,
                            cards,
                            replicas,
                            m,
                            n,
                            b_card,
                            microbatch_num,
                            beta,
                            path,
                            attn_weight,
                            expert_mem,
                            max_microbatch_size,
                            "network_bound_deep_pipeline",
                        )
                        if cand is None:
                            continue
                        if best_cand is None or cand.tpot_s < best_cand.tpot_s:
                            best_cand = cand
                    if best_cand is None:
                        continue
                    if math.isfinite(tau) and best_cand.tpot_s > tau:
                        continue
                    add_if_finite(candidates, best_cand)
                    seen_deep[b_card] = best_cand.tpot_s
                    break
    return candidates


def enumerate_collocated(
    model: ModelSpec,
    hw: Hardware,
    seq_len: int,
    kv_seq_len: float,
    total_cards: int,
    max_microbatch_size: int,
    beta: float,
    compact: bool,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    hbm = hw.hbm_capacity
    attn_weight = model.attn_weight_bytes
    moe_weight = model.moe_weight_bytes_total
    max_x = min(model.experts, total_cards)

    x_values = divisors_leq(total_cards, max_x) if compact else list(range(1, max_x + 1))
    for x in x_values:
        moe_mem = moe_weight / x
        path = comm_path_for_cards(x, hw)
        replicas = total_cards // x
        if replicas < 1:
            continue
        kv_room = hbm - attn_weight - moe_mem
        if kv_room <= 0:
            continue
        max_b_card_by_mem = int(kv_room // kv_cache_bytes(model, 1, kv_seq_len))
        if max_b_card_by_mem < 1:
            continue
        for microbatch_num in candidate_microbatch_nums(2, compact, afd=False):
            max_b_mu = min(max_microbatch_size, max_b_card_by_mem // microbatch_num)
            if max_b_mu < 1:
                continue
            for b_mu in candidate_batch_sizes(max_b_mu, compact):
                b_card = b_mu * microbatch_num
                routed_per_card = beta * model.top_k * b_mu
                t_attn_layer = attention_layer_time(model, hw, b_mu, seq_len)
                t_expert_layer = expert_layer_time(model, hw, routed_per_card, x)
                t_dispatch_layer = collocated_comm_time(
                    model, hw, x, b_mu, path
                )
                t_combine_layer = t_dispatch_layer

                attn_full = t_attn_layer * model.layers
                moe_full = t_expert_layer * model.layers
                dispatch_full = t_dispatch_layer * model.layers
                combine_full = t_combine_layer * model.layers
                compute_full = attn_full + moe_full
                comm_full = dispatch_full + combine_full

                if microbatch_num == 1:
                    t_iter = compute_full + comm_full
                    overlap = "none"
                else:
                    tbo_stage_1 = max(attn_full, combine_full)
                    tbo_stage_2 = max(dispatch_full, attn_full)
                    tbo_stage_3 = max(moe_full, dispatch_full)
                    tbo_stage_4 = max(combine_full, moe_full)
                    t_iter = tbo_stage_1 + tbo_stage_2 + tbo_stage_3 + tbo_stage_4
                    if combine_full <= attn_full and dispatch_full <= attn_full and dispatch_full <= moe_full and combine_full <= moe_full:
                        overlap = "two_batch_fully_hidden"
                    else:
                        overlap = "two_batch_partially_hidden"

                global_batch = x * b_card
                inst_throughput = global_batch / t_iter

                add_if_finite(
                    candidates,
                    Candidate(
                        mode="collocated",
                        tpot_s=t_iter,
                        throughput_tps=replicas * inst_throughput,
                        cards_used=replicas * x,
                        replicas=replicas,
                        instance_cards=x,
                        collocated_ep_width=x,
                        microbatch_size=b_mu,
                        microbatch_num=microbatch_num,
                        per_card_batch=b_card,
                        global_batch=replicas * global_batch,
                        memory_collocated_gib=memory_gib(
                            attn_weight
                            + moe_mem
                            + kv_cache_bytes(model, b_card, kv_seq_len)
                        ),
                        t_attn_ms=attn_full * 1e3,
                        t_moe_ms=moe_full * 1e3,
                        t_dispatch_ms=dispatch_full * 1e3,
                        t_combine_ms=combine_full * 1e3,
                        comm_path=path,
                        overlap=overlap,
                        extra={
                            "compute_full_ms": compute_full * 1e3,
                            "comm_full_ms": comm_full * 1e3,
                            "nominal_attention_seq_len": seq_len,
                            "kv_allocation_seq_len": kv_seq_len,
                            "tbo_stage_1_max_attn_prev_combine_ms": max(attn_full, combine_full) * 1e3 if microbatch_num == 2 else 0.0,
                            "tbo_stage_2_max_dispatch_attn_ms": max(dispatch_full, attn_full) * 1e3 if microbatch_num == 2 else 0.0,
                            "tbo_stage_3_max_moe_dispatch_ms": max(moe_full, dispatch_full) * 1e3 if microbatch_num == 2 else 0.0,
                            "tbo_stage_4_max_combine_moe_ms": max(combine_full, moe_full) * 1e3 if microbatch_num == 2 else 0.0,
                        },
                    ),
                )
    return candidates
