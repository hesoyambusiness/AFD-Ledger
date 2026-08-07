"""LongCat 2.0 analytical-versus-production architecture validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import load_hardware, resolve_hardware_config_path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "configs/models/longcat-2.0-validation.json"
MEASUREMENTS_PATH = ROOT / "configs/experiments/longcat_measurements.csv"
VALIDATION_CARD_BUDGET = 160

TP = 8
ATTN_BYTES = 2
KV_BYTES = 2
INT8_BYTES = 1
ACT_INT8_BYTES = 1
ROUTER_BYTES = 2
COMM_BYTES = 2


@dataclass(frozen=True)
class LongCatModel:
    layers: int
    hidden: int
    dense_hidden: int
    expert_hidden: int
    experts: int
    top_k: int
    heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int


def load_longcat_model() -> LongCatModel:
    with MODEL_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    data.pop("name", None)
    return LongCatModel(**data)


def roofline(flops: float, moved: float, peak: float, bandwidth: float) -> float:
    return max(flops / peak if flops else 0.0, moved / bandwidth if moved else 0.0)


def matmul(
    tokens: float,
    in_dim: float,
    out_dim: float,
    peak: float,
    bandwidth: float,
    weight_bytes: float,
    activation_bytes: float,
    weight_scale: float = 1.0,
) -> float:
    flops = 2.0 * tokens * in_dim * out_dim
    moved = (
        weight_scale * in_dim * out_dim * weight_bytes
        + tokens * (in_dim + out_dim) * activation_bytes
    )
    return roofline(flops, moved, peak, bandwidth)


def _index_entry(model: LongCatModel) -> int:
    return model.index_head_dim + ((model.index_head_dim + 127) // 128) * 4


def sparse_context(model: LongCatModel, peak: float, bandwidth: float, tokens: float) -> float:
    qk = model.kv_lora_rank + model.qk_rope_head_dim
    value = model.kv_lora_rank
    flops = 2.0 * tokens * model.index_topk * model.heads * (qk + value)
    moved = (
        tokens * model.index_topk * (qk + value) * KV_BYTES
        + tokens * model.heads * (qk + value) * ATTN_BYTES
    )
    return roofline(flops, moved, peak, bandwidth)


def attention_base(model: LongCatModel, peak: float, bandwidth: float, tokens: float) -> float:
    qk_head = model.qk_nope_head_dim + model.qk_rope_head_dim
    shapes = (
        (model.hidden, model.q_lora_rank),
        (model.q_lora_rank, model.heads * qk_head),
        (model.hidden, model.kv_lora_rank + model.qk_rope_head_dim),
        (model.kv_lora_rank, model.heads * model.qk_nope_head_dim),
        (model.heads * model.kv_lora_rank, model.hidden / TP),
    )
    times = [
        matmul(tokens, left, right, peak, bandwidth, ATTN_BYTES, ATTN_BYTES)
        for left, right in shapes
    ]
    times.insert(4, sparse_context(model, peak, bandwidth, tokens))
    return sum(times)


def attention_zero(
    model: LongCatModel,
    peak: float,
    bandwidth: float,
    tokens: float,
    seq_len: int,
) -> float:
    result = attention_base(model, peak, bandwidth, tokens)
    result += matmul(tokens, model.hidden, model.index_head_dim, peak, bandwidth, ATTN_BYTES, ATTN_BYTES)
    result += matmul(tokens, model.hidden, model.index_n_heads, peak, bandwidth, ATTN_BYTES, ATTN_BYTES)
    result += matmul(
        tokens,
        model.q_lora_rank,
        model.index_n_heads * model.index_head_dim,
        peak,
        bandwidth,
        ATTN_BYTES,
        ATTN_BYTES,
    )
    result += roofline(
        2.0 * tokens * seq_len * model.index_n_heads * model.index_head_dim,
        tokens * seq_len * _index_entry(model),
        peak,
        bandwidth,
    )
    return result


def dense_ffn(model: LongCatModel, peak: float, bandwidth: float, tokens: float) -> float:
    local = model.dense_hidden / TP
    return sum(
        (
            matmul(tokens, model.hidden, local, peak, bandwidth, INT8_BYTES, ACT_INT8_BYTES),
            matmul(tokens, model.hidden, local, peak, bandwidth, INT8_BYTES, ACT_INT8_BYTES),
            matmul(tokens, local, model.hidden, peak, bandwidth, INT8_BYTES, ACT_INT8_BYTES),
        )
    )


def router(model: LongCatModel, peak: float, bandwidth: float, tokens: float) -> float:
    return matmul(tokens, model.hidden, model.experts, peak, bandwidth, ROUTER_BYTES, ATTN_BYTES)


def experts(
    model: LongCatModel,
    peak: float,
    bandwidth: float,
    records: float,
    ep_width: int,
) -> float:
    scale = model.experts / ep_width
    return sum(
        (
            matmul(records, model.hidden, model.expert_hidden, peak, bandwidth, INT8_BYTES, ACT_INT8_BYTES, scale),
            matmul(records, model.hidden, model.expert_hidden, peak, bandwidth, INT8_BYTES, ACT_INT8_BYTES, scale),
            matmul(records, model.expert_hidden, model.hidden, peak, bandwidth, INT8_BYTES, ACT_INT8_BYTES, scale),
        )
    )


def communication(model: LongCatModel, bandwidth: float, tokens: float) -> float:
    return model.top_k * tokens * model.hidden * COMM_BYTES / bandwidth


def collocated_layer(model: LongCatModel, hw, seq_len: int, cards: int, batch: int):
    t0 = attention_zero(model, hw.peak_flops["bf16"], hw.hbm_bw, batch, seq_len)
    half_bw = hw.hbm_bw / 2.0
    main = (
        dense_ffn(model, hw.peak_flops["int8"] / 2.0, half_bw, batch)
        + attention_base(model, hw.peak_flops["bf16"] / 2.0, half_bw, batch)
        + dense_ffn(model, hw.peak_flops["int8"] / 2.0, half_bw, batch)
    )
    ep = min(cards, model.experts)
    records = model.top_k * batch * cards / ep
    moe = (
        router(model, hw.peak_flops["bf16"] / 2.0, half_bw, batch)
        + communication(model, hw.intra_node_bw, batch)
        + experts(model, hw.peak_flops["int8"] / 2.0, half_bw, records, ep)
        + communication(model, hw.intra_node_bw, batch)
    )
    return t0 + max(main, moe), (t0, main, moe)


def afd_layer(model: LongCatModel, hw, seq_len: int, attention_cards: int, expert_cards: int, batch: int):
    t0 = attention_zero(model, hw.peak_flops["bf16"], hw.hbm_bw, batch, seq_len)
    main = (
        dense_ffn(model, hw.peak_flops["int8"], hw.hbm_bw, batch)
        + attention_base(model, hw.peak_flops["bf16"], hw.hbm_bw, batch)
        + dense_ffn(model, hw.peak_flops["int8"], hw.hbm_bw, batch)
    )
    records = model.top_k * batch * attention_cards / expert_cards
    moe = (
        router(model, hw.peak_flops["bf16"], hw.hbm_bw, batch)
        + communication(model, hw.intra_node_bw, batch)
        + experts(model, hw.peak_flops["int8"], hw.hbm_bw, records, min(expert_cards, model.experts))
        + communication(model, hw.intra_node_bw, batch)
    )
    return t0 + max(main, moe), (t0, main, moe)


def best_split(model: LongCatModel, hw, budget: int, seq_len: int, batch: int):
    best: Optional[tuple] = None
    for expert_cards in range(32, budget):
        attention_cards = budget - expert_cards
        layer, parts = afd_layer(model, hw, seq_len, attention_cards, expert_cards, batch)
        throughput = attention_cards * batch / (layer * model.layers)
        candidate = (throughput, attention_cards, expert_cards, layer, parts)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError("no LongCat AFD split")
    return best[1], best[2], best[3], best[4]


def validation_rows() -> list[dict]:
    model = load_longcat_model()
    hw = load_hardware(resolve_hardware_config_path(Path("910C")))
    rows = []
    with MEASUREMENTS_PATH.open(newline="", encoding="utf-8") as handle:
        measurements = [
            row
            for row in csv.DictReader(handle)
            if int(row["card_budget"]) == VALIDATION_CARD_BUDGET
        ]
    for measured in measurements:
        budget = int(measured["card_budget"])
        seq_len = int(measured["seq_len"])
        collocated_cards = int(measured["collocated_cards"])
        collocated_batch = int(measured["collocated_batch_per_card"])
        collocated_s, collocated_parts = collocated_layer(
            model, hw, seq_len, collocated_cards, collocated_batch
        )
        if measured["search_best_afd"].lower() == "true":
            attention_cards, expert_cards, afd_s, afd_parts = best_split(
                model, hw, budget, seq_len, int(measured["afd_batch_per_attention_card"])
            )
        else:
            attention_cards = int(measured["afd_attention_cards"])
            expert_cards = int(measured["afd_expert_cards"])
            afd_s, afd_parts = afd_layer(
                model,
                hw,
                seq_len,
                attention_cards,
                expert_cards,
                int(measured["afd_batch_per_attention_card"]),
            )
        collocated_model_tps = (
            collocated_cards * collocated_batch / (collocated_s * model.layers)
        )
        collocated_model_tps *= budget / collocated_cards
        collocated_measured_tps = (
            float(measured["collocated_tps"]) * budget / collocated_cards
        )
        afd_model_tps = (
            attention_cards
            * int(measured["afd_batch_per_attention_card"])
            / (afd_s * model.layers)
        )
        model_ratio = afd_model_tps / collocated_model_tps
        measured_ratio = float(measured["afd_tps"]) / collocated_measured_tps
        rows.append(
            {
                **measured,
                "collocated_model_tps": collocated_model_tps,
                "collocated_model_tpot_ms": collocated_s * model.layers * 1e3,
                "afd_model_tps": afd_model_tps,
                "afd_model_tpot_ms": afd_s * model.layers * 1e3,
                "model_afd_attention_cards": attention_cards,
                "model_afd_expert_cards": expert_cards,
                "theta_model_ratio": model_ratio,
                "theta_measured_ratio": measured_ratio,
                "model_over_measured_ratio": model_ratio / measured_ratio,
                "collocated_t0_us": collocated_parts[0] * 1e6,
                "collocated_main_us": collocated_parts[1] * 1e6,
                "collocated_moe_us": collocated_parts[2] * 1e6,
                "afd_t0_us": afd_parts[0] * 1e6,
                "afd_main_us": afd_parts[1] * 1e6,
                "afd_moe_us": afd_parts[2] * 1e6,
                "measurement_provenance": "empirical input: Ascend 910C SuperPod",
            }
        )
    return rows
