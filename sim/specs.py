"""Core dataclasses shared by the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class Hardware:
    name: str
    hbm_bw: float
    hbm_capacity: float
    peak_flops: Mapping[str, float]
    intra_node_bw: float
    inter_node_bw: float
    pcie_bw: float
    gpus_per_node: int
    inter_rack_bw: Optional[float] = None
    gpus_per_rack: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "peak_flops",
            MappingProxyType(dict(self.peak_flops)),
        )

    def comm_bw(self, path: str) -> float:
        if path in {"intra", "nvlink", "hccs"}:
            return self.intra_node_bw
        if path in {"inter", "rack"}:
            return self.inter_node_bw
        if path in {"inter_rack", "rdma"}:
            return self.inter_rack_bw or self.inter_node_bw
        if path == "pcie":
            return self.pcie_bw
        raise ValueError(f"unknown communication path: {path}")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    hidden: int
    moe_hidden: int
    experts: int
    top_k: int
    heads: int
    kv_heads: int
    head_dim: int
    weight_precision: str
    compute_precision: str
    attention_precision: str
    weight_bytes: int
    router_precision: str
    router_weight_bytes: int
    act_bytes: int
    kv_bytes: int
    attention_kind: str = "gqa"
    kv_dim_override: Optional[float] = None
    dsa: Optional[Mapping[str, float]] = None

    def __post_init__(self) -> None:
        if self.dsa is not None:
            object.__setattr__(self, "dsa", MappingProxyType(dict(self.dsa)))

    @property
    def kv_dim(self) -> float:
        if self.kv_dim_override is not None:
            return self.kv_dim_override
        return self.kv_heads * self.head_dim

    @property
    def attn_params_per_layer(self) -> int:
        if self.attention_kind == "dsa":
            if self.dsa is None:
                raise ValueError("DSA model is missing DSA geometry")
            q_lora = int(self.dsa["q_lora_rank"])
            kv_lora = int(self.dsa["kv_lora_rank"])
            qk_nope = int(self.dsa["qk_nope_head_dim"])
            qk_rope = int(self.dsa["qk_rope_head_dim"])
            v_dim = int(self.dsa["v_head_dim"])
            index_head_dim = int(self.dsa["index_head_dim"])
            index_n_heads = int(self.dsa["index_n_heads"])
            proj_out = q_lora + kv_lora + qk_rope + index_head_dim
            qk_head_dim = qk_nope + qk_rope
            dsa_attn = (
                self.hidden * proj_out
                + q_lora * self.heads * qk_head_dim
                + q_lora * index_n_heads * index_head_dim
                + self.hidden * index_n_heads
                + self.heads * v_dim * self.hidden
            )
            return dsa_attn

        return int(2 * self.hidden * self.hidden + 2 * self.hidden * self.kv_dim)

    @property
    def router_params_per_layer(self) -> int:
        return self.hidden * self.experts

    @property
    def moe_params_per_expert_per_layer(self) -> int:
        # SwiGLU experts: gate, up, down projections.
        return 3 * self.hidden * self.moe_hidden

    @property
    def attn_weight_bytes(self) -> float:
        return self.layers * (
            self.attn_params_per_layer * self.weight_bytes
            + self.router_params_per_layer * self.router_weight_bytes
        )

    @property
    def moe_weight_bytes_total(self) -> float:
        return (
            self.layers
            * self.experts
            * self.moe_params_per_expert_per_layer
            * self.weight_bytes
        )


@dataclass
class Candidate:
    mode: str
    tpot_s: float
    throughput_tps: float
    cards_used: int
    replicas: int
    instance_cards: int
    m: Optional[int] = None
    n: Optional[int] = None
    collocated_ep_width: Optional[int] = None
    microbatch_size: int = 0
    microbatch_num: int = 0
    per_card_batch: int = 0
    global_batch: int = 0
    memory_attn_gib: float = 0.0
    memory_expert_gib: float = 0.0
    memory_collocated_gib: float = 0.0
    t_attn_ms: float = 0.0
    t_moe_ms: float = 0.0
    t_dispatch_ms: float = 0.0
    t_combine_ms: float = 0.0
    comm_path: str = ""
    overlap: str = ""
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def tpot_ms(self) -> float:
        return self.tpot_s * 1e3

    @property
    def generation_speed_tps_per_user(self) -> float:
        return 1.0 / self.tpot_s
