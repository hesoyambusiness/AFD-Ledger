"""Standalone JSON config loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .precision import dtype_bytes, normalize_precision
from .specs import Hardware, ModelSpec


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
MODEL_CONFIG_DIR = CONFIG_DIR / "models"
HARDWARE_CONFIG_DIR = CONFIG_DIR / "hardware"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f, parse_constant=lambda _value: None)


def _load_structured(path: Path) -> dict:
    return _load_json(path)


def _model_id_from_config_path(path: Path) -> str:
    name = path.name
    if name.endswith("_config.json"):
        name = name[: -len("_config.json")]
    return name.replace("--", "/")


def _model_config_candidates():
    for candidate in MODEL_CONFIG_DIR.glob("*.json"):
        if candidate.name.endswith("_hf_quant_config.json"):
            continue
        yield candidate


def resolve_model_config_path(path_or_id: Path) -> Path:
    if path_or_id.exists():
        return path_or_id

    value = str(path_or_id)
    direct_candidates = [
        MODEL_CONFIG_DIR / value,
        MODEL_CONFIG_DIR / f"{value}.json",
        MODEL_CONFIG_DIR / f"{value.replace('/', '--')}_config.json",
    ]
    for direct in direct_candidates:
        if direct.exists():
            return direct

    normalized = value.lower()
    exact = []
    partial = []
    for candidate in _model_config_candidates():
        model_id = _model_id_from_config_path(candidate)
        model_name = model_id.rsplit("/", 1)[-1]
        stem = candidate.stem
        if normalized in {model_id.lower(), model_name.lower(), stem.lower()}:
            exact.append(candidate)
        elif normalized in model_id.lower() or normalized in model_name.lower() or normalized in stem.lower():
            partial.append(candidate)
    matches = exact or partial
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(_model_id_from_config_path(p) for p in matches)
        raise ValueError(f"model id {value!r} is ambiguous; matches: {names}")

    raise FileNotFoundError(f"model config not found: {value}")


def resolve_hardware_config_path(path_or_name: Path) -> Path:
    if path_or_name.exists():
        return path_or_name

    value = str(path_or_name)
    direct_candidates = [
        HARDWARE_CONFIG_DIR / value,
        HARDWARE_CONFIG_DIR / f"{value}.json",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"hardware config not found: {value}")


def _model_name(path: Path) -> str:
    if path.name.endswith("_config.json"):
        return _model_id_from_config_path(path)
    return path.stem


def _merge_hf_quant_sidecar(path: Path, data: dict) -> dict:
    if data.get("hf_quant_config"):
        return data
    sidecar_name = path.name.removesuffix("_config.json") + "_hf_quant_config.json"
    sidecar = path.with_name(sidecar_name)
    if not sidecar.exists():
        return data
    merged = dict(data)
    merged["hf_quant_config"] = _load_json(sidecar)
    return merged


def _maybe_text_config(data: dict) -> dict:
    architecture = (data.get("architectures") or [""])[0]
    text_config_keys = {
        "KimiK25ForConditionalGeneration": "text_config",
        "Llama4ForConditionalGeneration": "text_config",
        "Qwen3_5ForConditionalGeneration": "text_config",
        "Qwen3_5MoeForConditionalGeneration": "text_config",
        "Gemma4ForConditionalGeneration": "text_config",
        "Qwen3VLForConditionalGeneration": "text_config",
        "Qwen3VLMoeForConditionalGeneration": "text_config",
    }
    text_key = text_config_keys.get(architecture)
    if text_key and isinstance(data.get(text_key), dict):
        text = dict(data[text_key])
        text["architectures"] = [architecture]
        if "quantization_config" not in text and "quantization_config" in data:
            text["quantization_config"] = data["quantization_config"]
        if "hf_quant_config" not in text and "hf_quant_config" in data:
            text["hf_quant_config"] = data["hf_quant_config"]
        return text
    return data


def _first_positive_int(data: dict, *keys: str) -> int:
    for key in keys:
        value = data.get(key)
        if value is not None and int(value) > 0:
            return int(value)
    raise ValueError(f"model config must define one of: {', '.join(keys)}")


def _weight_precision(data: dict) -> str:
    quant = data.get("quantization_config") or {}
    hf_quant = data.get("hf_quant_config") or {}
    hf_quant_section = hf_quant.get("quantization") or {}
    precision = normalize_precision(
        quant.get("quant_algo")
        or hf_quant_section.get("quant_algo")
        or quant.get("quant_method")
        or data.get("torch_dtype")
        or data.get("dtype")
        or "bf16"
    )
    if precision == "modelopt":
        precision = normalize_precision(hf_quant_section.get("quant_algo") or "fp8")
    return precision


def _dsa_kv_dim_override(data: dict, kv_bytes: int) -> float:
    kv_lora = float(data["kv_lora_rank"])
    qk_rope = float(data["qk_rope_head_dim"])
    index_head_dim = int(data["index_head_dim"])
    indexer_entry_bytes = index_head_dim + ((index_head_dim + 127) // 128) * 4
    per_token_layer_bytes = kv_lora * kv_bytes + qk_rope * dtype_bytes("bf16") + indexer_entry_bytes
    return per_token_layer_bytes / (2.0 * kv_bytes)


def load_hardware(path: Path) -> Hardware:
    data = _load_structured(path)

    gpu = data["gpu"]
    node = data["node"]
    peak = {}
    if "fp8_tc_flops" in gpu:
        peak["fp8"] = float(gpu["fp8_tc_flops"])
    if "int8_tc_flops" in gpu:
        peak["int8"] = float(gpu["int8_tc_flops"])
    if "fp4_tc_flops" in gpu:
        peak["fp4"] = float(gpu["fp4_tc_flops"])
        peak["nvfp4"] = float(gpu["fp4_tc_flops"])
    if "bfloat16_tc_flops" in gpu:
        peak["bf16"] = float(gpu["bfloat16_tc_flops"])
        peak["fp16"] = float(gpu["bfloat16_tc_flops"])

    return Hardware(
        name=path.stem,
        hbm_bw=float(gpu["mem_bw"]),
        hbm_capacity=float(gpu["mem_capacity"]),
        peak_flops=peak,
        intra_node_bw=float(node["intra_node_bw"]),
        inter_node_bw=float(node["inter_node_bw"]),
        pcie_bw=float(node.get("pcie_bw", node["inter_node_bw"])),
        gpus_per_node=int(node["num_gpus_per_node"]),
        inter_rack_bw=float(node["inter_rack_bw"]) if "inter_rack_bw" in node else None,
        gpus_per_rack=int(node["num_gpus_per_rack"]) if "num_gpus_per_rack" in node else None,
    )


def load_model(
    path: Path,
    compute_precision: Optional[str],
    attention_precision: str,
    activation_precision: str,
    kv_precision: str,
) -> ModelSpec:
    path = resolve_model_config_path(path)
    data = _maybe_text_config(_merge_hf_quant_sidecar(path, _load_json(path)))

    architecture = (data.get("architectures") or [""])[0]
    experts = _first_positive_int(data, "num_experts", "n_routed_experts", "num_local_experts")
    top_k = _first_positive_int(data, "num_experts_per_tok", "num_experts_per_token", "top_k_experts", "experts_per_token")
    weight_precision = _weight_precision(data)

    selected_compute = normalize_precision(compute_precision or weight_precision)
    selected_attention = normalize_precision(attention_precision)
    act_precision = normalize_precision(activation_precision)
    kv_precision = normalize_precision(kv_precision)
    kv_bytes = dtype_bytes(kv_precision)
    act_bytes = dtype_bytes(act_precision)

    attention_kind = "dsa" if architecture == "DeepseekV32ForCausalLM" else "gqa"
    dsa = None
    kv_dim_override = None
    if attention_kind == "dsa":
        required = [
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "index_head_dim",
            "index_n_heads",
            "index_topk",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"DeepSeek-V3.2 DSA config missing required fields: {', '.join(missing)}")
        dsa = {key: float(data[key]) for key in required}
        kv_dim_override = _dsa_kv_dim_override(data, kv_bytes)

    return ModelSpec(
        name=_model_name(path),
        layers=int(data["num_hidden_layers"]),
        hidden=int(data["hidden_size"]),
        moe_hidden=int(data.get("moe_intermediate_size") or data["intermediate_size"]),
        experts=experts,
        top_k=top_k,
        heads=int(data["num_attention_heads"]),
        kv_heads=int(data.get("num_key_value_heads", data["num_attention_heads"])),
        head_dim=int(data.get("head_dim", data["hidden_size"] // data["num_attention_heads"])),
        weight_precision=weight_precision,
        compute_precision=selected_compute,
        attention_precision=selected_attention,
        weight_bytes=dtype_bytes(weight_precision),
        router_precision="bf16",
        router_weight_bytes=dtype_bytes("bf16"),
        act_bytes=act_bytes,
        kv_bytes=kv_bytes,
        attention_kind=attention_kind,
        kv_dim_override=kv_dim_override,
        dsa=dsa,
    )
