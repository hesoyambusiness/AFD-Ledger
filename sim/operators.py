"""Current roofline formulas for GQA attention, routed MoE, and communication."""

from __future__ import annotations

import math
from typing import Optional

from .precision import resolve_compute_precision
from .specs import Hardware, ModelSpec


def roofline_time(flops: float, bytes_moved: float, precision: str, hw: Hardware) -> float:
    arithmetic_precision = resolve_compute_precision(precision, hw.peak_flops)
    compute_s = flops / hw.peak_flops[arithmetic_precision] if flops > 0 else 0.0
    memory_s = bytes_moved / hw.hbm_bw if bytes_moved > 0 else 0.0
    return max(compute_s, memory_s)


def compute_time(flops: float, precision: str, hw: Hardware) -> float:
    return roofline_time(flops, 0.0, precision, hw)


def matmul_time(
    tokens: float,
    in_dim: float,
    out_dim: float,
    precision: str,
    hw: Hardware,
    weight_bytes: float,
    act_bytes: float,
    weight_scale: float = 1.0,
) -> float:
    flops = 2.0 * tokens * in_dim * out_dim
    bytes_moved = weight_scale * in_dim * out_dim * weight_bytes + tokens * (in_dim + out_dim) * act_bytes
    return roofline_time(flops, bytes_moved, precision, hw)


def indexer_cache_entry_bytes(index_head_dim: int) -> float:
    return index_head_dim + ((index_head_dim + 127) // 128) * 4


def kv_cache_bytes(model: ModelSpec, batch: int, seq_len: float) -> float:
    if model.attention_kind == "dsa":
        if model.dsa is None:
            raise ValueError("DSA model is missing DSA geometry")
        kv_lora = float(model.dsa["kv_lora_rank"])
        qk_rope = float(model.dsa["qk_rope_head_dim"])
        index_head_dim = int(model.dsa["index_head_dim"])
        per_token_layer = (
            kv_lora * model.kv_bytes
            + qk_rope * 2.0
            + indexer_cache_entry_bytes(index_head_dim)
        )
        return model.layers * batch * seq_len * per_token_layer

    return 2.0 * model.kv_bytes * model.layers * batch * seq_len * model.kv_dim


def attention_layer_time(
    model: ModelSpec,
    hw: Hardware,
    b: int,
    seq_len: float,
    dsa_sparse_seq_len: Optional[float] = None,
) -> float:
    if model.attention_kind == "dsa":
        return dsa_attention_layer_time(model, hw, b, seq_len, dsa_sparse_seq_len)

    h = model.hidden
    kv = model.kv_dim

    qkv_t = matmul_time(
        b,
        h,
        h + 2.0 * kv,
        model.compute_precision,
        hw,
        model.weight_bytes,
        model.act_bytes,
    )

    attn_flops = 4.0 * b * seq_len * h
    attn_bytes = 2.0 * b * seq_len * kv * model.kv_bytes + b * h * model.act_bytes
    attn_t = roofline_time(attn_flops, attn_bytes, model.attention_precision, hw)

    out_t = matmul_time(b, h, h, model.compute_precision, hw, model.weight_bytes, model.act_bytes)

    router_flops = 2.0 * b * h * model.experts
    router_bytes = h * model.experts * model.router_weight_bytes + b * (h + model.experts) * model.act_bytes
    router_t = roofline_time(router_flops, router_bytes, model.router_precision, hw)

    return qkv_t + attn_t + out_t + router_t


def dsa_attention_layer_time(
    model: ModelSpec,
    hw: Hardware,
    b: int,
    seq_len: float,
    sparse_seq_len: Optional[float] = None,
) -> float:
    if model.dsa is None:
        raise ValueError("DSA model is missing DSA geometry")

    dsa = model.dsa
    h = model.hidden
    heads = model.heads
    q_lora = float(dsa["q_lora_rank"])
    kv_lora = float(dsa["kv_lora_rank"])
    qk_nope = float(dsa["qk_nope_head_dim"])
    qk_rope = float(dsa["qk_rope_head_dim"])
    v_dim = float(dsa["v_head_dim"])
    index_head_dim = float(dsa["index_head_dim"])
    index_n_heads = float(dsa["index_n_heads"])
    index_topk = float(dsa["index_topk"])

    qk_head_dim = qk_nope + qk_rope
    attn_head_dim = kv_lora + qk_rope
    sparse_window = seq_len if sparse_seq_len is None else sparse_seq_len
    effective_kv = min(sparse_window, index_topk)
    proj_out = q_lora + kv_lora + qk_rope + index_head_dim

    qkv_a_t = matmul_time(
        b,
        h,
        proj_out,
        model.compute_precision,
        hw,
        model.weight_bytes,
        model.act_bytes,
    )
    q_b_t = matmul_time(
        b,
        q_lora,
        heads * qk_head_dim,
        model.compute_precision,
        hw,
        model.weight_bytes,
        model.act_bytes,
    )
    index_query_t = matmul_time(
        b,
        q_lora,
        index_n_heads * index_head_dim,
        model.compute_precision,
        hw,
        model.weight_bytes,
        model.act_bytes,
    )
    index_weight_t = matmul_time(
        b,
        h,
        index_n_heads,
        model.compute_precision,
        hw,
        model.weight_bytes,
        model.act_bytes,
    )
    out_t = matmul_time(
        b,
        heads * v_dim,
        h,
        model.compute_precision,
        hw,
        model.weight_bytes,
        model.act_bytes,
    )
    q_w_kc_t = roofline_time(
        2.0 * heads * b * qk_nope * kv_lora,
        b * heads * (qk_nope + kv_lora) * model.act_bytes,
        model.compute_precision,
        hw,
    )
    s_w_vc_t = roofline_time(
        2.0 * heads * b * kv_lora * v_dim,
        b * heads * (kv_lora + v_dim) * model.act_bytes,
        model.compute_precision,
        hw,
    )

    indexer_logits_flops = 2.0 * b * index_n_heads * index_head_dim * seq_len
    indexer_entry = indexer_cache_entry_bytes(int(index_head_dim))
    indexer_cache_bytes = b * seq_len * indexer_entry
    indexer_logits_t = roofline_time(
        indexer_logits_flops,
        indexer_cache_bytes,
        model.compute_precision,
        hw,
    )

    sparse_attn_flops = 2.0 * b * heads * (attn_head_dim + kv_lora) * effective_kv
    kv_cache_read_bytes = b * effective_kv * attn_head_dim * model.kv_bytes
    sparse_attn_t = roofline_time(
        sparse_attn_flops,
        kv_cache_read_bytes + b * heads * (qk_head_dim + v_dim) * model.act_bytes,
        "bf16",
        hw,
    )

    router_flops = 2.0 * b * h * model.experts
    router_bytes = h * model.experts * model.router_weight_bytes + b * (h + model.experts) * model.act_bytes
    router_t = roofline_time(router_flops, router_bytes, model.router_precision, hw)

    return (
        qkv_a_t
        + q_b_t
        + index_query_t
        + index_weight_t
        + out_t
        + q_w_kc_t
        + s_w_vc_t
        + indexer_logits_t
        + sparse_attn_t
        + router_t
    )


def runtime_attention_layer_time(
    model: ModelSpec,
    hw: Hardware,
    b: int,
    nominal_seq_len: int,
    runtime_seq_len: float,
) -> float:
    if model.attention_kind == "dsa":
        return attention_layer_time(
            model,
            hw,
            b,
            runtime_seq_len,
            dsa_sparse_seq_len=nominal_seq_len,
        )
    return attention_layer_time(model, hw, b, runtime_seq_len)


def expert_layer_time(
    model: ModelSpec,
    hw: Hardware,
    routed_tokens_per_card: float,
    ep_width: int,
) -> float:
    h = model.hidden
    d = model.moe_hidden

    expert_weight_scale = model.experts / ep_width
    gate_t = matmul_time(
        routed_tokens_per_card,
        h,
        d,
        model.compute_precision,
        hw,
        model.weight_bytes,
        model.act_bytes,
        expert_weight_scale,
    )
    up_t = matmul_time(
        routed_tokens_per_card,
        h,
        d,
        model.compute_precision,
        hw,
        model.weight_bytes,
        model.act_bytes,
        expert_weight_scale,
    )
    down_t = matmul_time(
        routed_tokens_per_card,
        d,
        h,
        model.compute_precision,
        hw,
        model.weight_bytes,
        model.act_bytes,
        expert_weight_scale,
    )
    return gate_t + up_t + down_t


def comm_path_for_cards(cards: int, hw: Hardware) -> str:
    if cards <= hw.gpus_per_node:
        return "intra"
    if hw.gpus_per_rack is not None and cards <= hw.gpus_per_rack:
        return "inter"
    return "rdma"


def dispatch_or_combine_time(
    model: ModelSpec,
    hw: Hardware,
    source_cards: int,
    expert_cards: int,
    b_mu: int,
    path: str,
) -> float:
    sender_payload, receiver_payload, payload = dispatch_or_combine_payloads(
        model,
        source_cards,
        expert_cards,
        b_mu,
        path,
    )
    return payload / hw.comm_bw(path)


def rdma_push_pull_path(path: str) -> bool:
    return path in {"inter", "rack", "inter_rack", "rdma"}


def dispatch_or_combine_payloads(
    model: ModelSpec,
    source_cards: int,
    expert_cards: int,
    b_mu: int,
    path: str,
) -> tuple[float, float, float]:
    attention_side_payload = model.top_k * b_mu * model.hidden * model.act_bytes
    expert_side_tokens = math.ceil(source_cards * b_mu / expert_cards)
    expert_side_payload = model.top_k * expert_side_tokens * model.hidden * model.act_bytes
    selected_payload = min(attention_side_payload, expert_side_payload)
    return attention_side_payload, expert_side_payload, selected_payload


def collocated_comm_time(
    model: ModelSpec,
    hw: Hardware,
    ep_width: int,
    microbatch: int,
    path: str,
) -> float:
    payload = model.top_k * microbatch * model.hidden * model.act_bytes
    return payload / hw.comm_bw(path)


def memory_gib(bytes_value: float) -> float:
    return bytes_value / (1024**3)
