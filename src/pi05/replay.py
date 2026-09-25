"""Native pi0.5 block replay and cached call-state loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_rows(value: torch.Tensor, limit: int) -> torch.Tensor:
    value = value.reshape(-1, value.shape[-1])
    if value.shape[0] > limit:
        indices = torch.linspace(0, value.shape[0] - 1, limit, device=value.device).round().long()
        value = value.index_select(0, indices)
    return value.detach()


@dataclass
class VisionReplayState:
    hidden: torch.Tensor
    attention_mask: torch.Tensor | None


@dataclass
class ReplayState:
    action_input: torch.Tensor
    time_input: torch.Tensor
    hidden_reference: torch.Tensor
    cond_reference: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    prefix_keys: list[torch.Tensor]
    prefix_values: list[torch.Tensor]
    hidden: torch.Tensor | None = None
    cond: torch.Tensor | None = None
    vision: list[VisionReplayState] | None = None
    language_hidden_reference: torch.Tensor | None = None
    language_attention_mask: torch.Tensor | None = None
    language_position_ids: torch.Tensor | None = None
    language_hidden: torch.Tensor | None = None
    flow_index: int | None = None
    request_index: int | None = None
    prompt_signature: int | None = None


def load_replay_states(
    tensor_path: Path,
    manifest_path: Path,
    expected_task: str,
    device: torch.device,
    max_states: int = 0,
    required_flow_index: int | None = None,
) -> tuple[list[ReplayState], dict[str, Any]]:
    manifest = load_json(manifest_path)
    if manifest.get("task") != expected_task:
        raise ValueError(f"Expected {expected_task} calibration, got {manifest.get('task')}")
    count = int(manifest["sample_count"])
    if max_states < 0:
        raise ValueError("max_states must be nonnegative")
    source_indices = list(range(count))
    sample_metadata = manifest.get("samples")
    if required_flow_index is not None:
        if not isinstance(sample_metadata, list) or len(sample_metadata) != count:
            raise ValueError("Flow-index filtering requires aligned manifest sample metadata")
        source_indices = [
            index
            for index, sample in enumerate(sample_metadata)
            if int(sample.get("flow_index", -1)) == required_flow_index
        ]
        if not source_indices:
            raise ValueError(f"No replay states found for flow_index={required_flow_index}")
    if max_states and len(source_indices) > max_states:
        source_count = len(source_indices)
        selected_offsets = (
            torch.linspace(0, source_count - 1, max_states).round().to(dtype=torch.int64).tolist()
        )
        source_indices = [source_indices[offset] for offset in selected_offsets]
    states: list[ReplayState] = []
    with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
        for index in source_indices:
            prefix = f"sample_{index:03d}."
            get = lambda name: tensors.get_tensor(prefix + name).to(device)  # noqa: E731
            full_prefix = manifest.get("method") == (
                "pi05_full_vision_language_action_block_regmeanpp_replay_calibration"
            )
            vision: list[VisionReplayState] | None = None
            language_hidden_reference = None
            language_attention_mask = None
            language_position_ids = None
            if full_prefix:
                vision_count = int(tensors.get_tensor(prefix + "vision_count"))
                vision = []
                keys = set(tensors.keys())
                for vision_index in range(vision_count):
                    vision_prefix = prefix + f"vision_{vision_index:02d}."
                    attention_key = vision_prefix + "attention_mask"
                    vision.append(
                        VisionReplayState(
                            hidden=tensors.get_tensor(vision_prefix + "hidden").to(device),
                            attention_mask=(
                                tensors.get_tensor(attention_key).to(device)
                                if attention_key in keys
                                else None
                            ),
                        )
                    )
                language_hidden_reference = get("language_hidden_mean")
                language_attention_mask = get("language_attention_mask")
                language_position_ids = get("language_position_ids")
            states.append(
                ReplayState(
                    action_input=get("action_input"),
                    time_input=get("time_input"),
                    hidden_reference=get("hidden_mean"),
                    cond_reference=get("cond_mean"),
                    attention_mask=get("attention_mask"),
                    position_ids=get("position_ids"),
                    prefix_keys=[get(f"prefix_key_{layer:02d}") for layer in range(18)],
                    prefix_values=[get(f"prefix_value_{layer:02d}") for layer in range(18)],
                    vision=vision,
                    language_hidden_reference=language_hidden_reference,
                    language_attention_mask=language_attention_mask,
                    language_position_ids=language_position_ids,
                    flow_index=(
                        int(sample_metadata[index]["flow_index"])
                        if isinstance(sample_metadata, list)
                        and sample_metadata[index].get("flow_index") is not None
                        else None
                    ),
                    request_index=(
                        int(sample_metadata[index]["request_index"])
                        if isinstance(sample_metadata, list)
                        and sample_metadata[index].get("request_index") is not None
                        else None
                    ),
                    prompt_signature=(
                        int(sample_metadata[index]["prompt_signature"])
                        if isinstance(sample_metadata, list)
                        and sample_metadata[index].get("prompt_signature") is not None
                        else None
                    ),
                )
            )
    return states, manifest


def resolve_module(modules: dict[str, torch.nn.Module], suffix: str) -> torch.nn.Module:
    if suffix in modules:
        return modules[suffix]
    matches = [module for name, module in modules.items() if name.endswith(suffix)]
    if len(matches) != 1:
        names = [name for name in modules if name.endswith(suffix)]
        raise KeyError(f"Could not uniquely resolve {suffix!r}: {names}")
    return matches[0]


def linear_forward(module: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    weight = module.weight
    if value.dtype != weight.dtype:
        value = value.to(weight.dtype)
    return F.linear(value, weight, module.bias)


def manual_action_block(
    layer: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    prefix_key: torch.Tensor | None,
    prefix_value: torch.Tensor | None,
    cond: torch.Tensor | None,
) -> torch.Tensor:
    """Run one PiGemma decoder layer without mutating a DynamicCache."""
    from lerobot.policies.pi_gemma import _gated_residual, layernorm_forward
    from transformers.models.gemma import modeling_gemma

    residual = hidden
    normalized, gate = layernorm_forward(layer.input_layernorm, hidden, cond)
    input_shape = normalized.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    value = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)
    cos, sin = rotary_emb(normalized, position_ids)
    query, key = modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin)
    if prefix_key is not None or prefix_value is not None:
        if prefix_key is None or prefix_value is None:
            raise ValueError("prefix_key and prefix_value must either both be set or both be None")
        key = torch.cat((prefix_key.to(dtype=key.dtype), key), dim=-2)
        value = torch.cat((prefix_value.to(dtype=value.dtype), value), dim=-2)
    attn_output, _ = modeling_gemma.eager_attention_forward(
        layer.self_attn,
        query,
        key,
        value,
        attention_mask,
        scaling=layer.self_attn.scaling,
        dropout=0.0,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = layer.self_attn.o_proj(attn_output)
    hidden = _gated_residual(residual, attn_output, gate)
    residual = hidden
    normalized, gate = layernorm_forward(layer.post_attention_layernorm, hidden, cond)
    if normalized.dtype != layer.mlp.up_proj.weight.dtype:
        normalized = normalized.to(layer.mlp.up_proj.weight.dtype)
    hidden = layer.mlp(normalized)
    return _gated_residual(residual, hidden, gate)


def decoder_prefix_kv(
    layer: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    cond: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact post-RoPE K/V tensors cached by a Gemma prefix layer."""
    from lerobot.policies.pi_gemma import layernorm_forward
    from transformers.models.gemma import modeling_gemma

    normalized, _ = layernorm_forward(layer.input_layernorm, hidden, cond)
    input_shape = normalized.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    value = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)
    cos, sin = rotary_emb(normalized, position_ids)
    _, key = modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin)
    return key, value


def augmented(weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.cat((weight.float(), bias.float().unsqueeze(1)), dim=1)


def with_ones(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return torch.cat((value, torch.ones((*value.shape[:-1], 1), device=value.device)), dim=-1)
