"""Sequential pi0.5 replay, preserving the production block computation order."""

from __future__ import annotations

import json
import math
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tcr_merging.calibration.traces import validate_rows
from tcr_merging.config import validate_inputs
from tcr_merging.merging.prefix import advance_by_prefix
from tcr_merging.merging.regression import solve_weight_multi
from tcr_merging.merging.sampling import row_indices
from tcr_merging.pi05.checkpoints import copy_support, sha256
from tcr_merging.pi05.replay import (
    ReplayState,
    augmented,
    decoder_prefix_kv,
    linear_forward,
    load_replay_states,
    manual_action_block,
    resolve_module,
    sample_rows,
    with_ones,
)
from tcr_merging.pi05.scope import FRONTEND_MODULES, OUTPUT_MODULE, module_groups, target_keys


@dataclass
class LinearSpec:
    adapter_name: str
    base_name: str
    weight_key: str
    module: torch.nn.Module
    weights: dict[str, torch.Tensor]
    prior: torch.Tensor


def concatenate_grouped(values, labels):
    if len(values) != len(labels):
        raise ValueError("Values/labels differ")
    groups = {}
    for value, label in zip(values, labels, strict=True):
        groups.setdefault(label, []).append(value)
    return {key: torch.cat(group, dim=0) for key, group in groups.items()}


def load_ridges(path, expert_hashes):
    path = Path(path)
    manifest = json.loads((path / "tcr_manifest.json").read_text())
    if manifest["pass_index"] != 1 or manifest["expert_sha256"] != expert_hashes:
        raise ValueError("Ridge reference must be pass 1 with the same experts")
    if sha256(path / "model.safetensors") != manifest["model_sha256"]:
        raise ValueError("Ridge-reference checkpoint hash changed")
    ridges = {name: float(entry["ridge"]) for name, entry in manifest["modules"].items()}
    if len(ridges) != 418 or not all(math.isfinite(v) and v > 0 for v in ridges.values()):
        raise ValueError("Ridge reference must cover all 418 modules")
    return ridges, manifest


@torch.inference_mode()
def merge_pass(cfg, pass_index, *, backend=None):
    """Build one pass; callers run passes in separate processes to release memory."""
    validate_inputs(cfg)
    if pass_index not in (1, 2) or pass_index > cfg["passes"]:
        raise ValueError("Invalid pass index")
    args = SimpleNamespace(**cfg)
    args.pass_index = pass_index
    args.ridge_scale = "feature_energy"
    args.expert_loss_normalization = cfg["first_pass_weighting"] if pass_index == 1 else "none"
    args.replay_prefix = "expert" if cfg["variant"] == "expert_prefix" else "merged"
    if backend is not None:
        if pass_index != 1 or cfg["passes"] != 1 or cfg.get("ridge_reference"):
            raise ValueError("Baselines use one independent pass, not TCR ridge inheritance")
        args.replay_prefix = "expert" if backend.teacher_only else "merged"
    args.final_call_only = cfg["variant"] == "final_call"
    experts = {name: Path(e["checkpoint"]) for name, e in cfg["experts"].items()}
    names = list(experts)
    base_root = Path(cfg["base_model"])
    output = (
        backend.output
        if backend is not None
        else Path(cfg["output"]) / ("pass1" if pass_index == 1 else "merged")
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    prior_root = Path(cfg["output"]) / "pass1" if pass_index == 2 else None
    print(f"Pass {pass_index}: checking expert/cache identities", flush=True)
    expert_hashes = {n: sha256(p / "model.safetensors") for n, p in experts.items()}
    cache_key = "cache_a" if pass_index == 1 else "cache_b"
    cache_hashes = {
        n: sha256(Path(e[cache_key]) / "replay.safetensors") for n, e in cfg["experts"].items()
    }
    full_ridges = {}
    ridge_root = Path(cfg["ridge_reference"]) if cfg.get("ridge_reference") else prior_root
    if ridge_root is not None:
        full_ridges, ridge_manifest = load_ridges(ridge_root, expert_hashes)
        if cfg.get("ridge_reference"):
            if ridge_manifest["config"]["variant"] != "full":
                raise ValueError("Ablations require a matching Full ridge reference")
            if cfg["variant"] in ("final_call", "expert_prefix"):
                reference_cache = "cache_a" if pass_index == 1 else "cache_b"
                expected = {
                    n: sha256(Path(e[reference_cache]) / "replay.safetensors")
                    for n, e in ridge_manifest["config"]["experts"].items()
                }
                if expected != cache_hashes:
                    raise ValueError(
                        "Generation/prefix ablations must use the matching Full raw caches"
                    )
    if prior_root is not None:
        _, parent = load_ridges(prior_root, expert_hashes)
        if parent["config"] != cfg:
            raise ValueError("Pass 2 must start from pass 1 of this exact configuration")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    states_by_name, manifests_by_name = {}, {}
    for name, entry in cfg["experts"].items():
        cache = Path(entry[cache_key])
        raw_manifest = json.loads((cache / "replay.json").read_text())
        if raw_manifest.get("model_sha256") and raw_manifest["model_sha256"] != expert_hashes[name]:
            raise ValueError(f"{name}: cached teacher hash differs from supplied expert")
        if (
            raw_manifest.get("tensor_sha256")
            and raw_manifest["tensor_sha256"] != cache_hashes[name]
        ):
            raise ValueError(f"{name}: cached replay tensor hash differs from manifest")
        states, manifest = load_replay_states(
            cache / "replay.safetensors",
            cache / "replay.json",
            name,
            device,
            required_flow_index=9 if args.final_call_only else None,
        )
        if args.final_call_only:
            manifest = {
                **manifest,
                "samples": [s for s in manifest["samples"] if s["flow_index"] == 9],
            }
        states_by_name[name] = states
        manifests_by_name[name] = [manifest]
    state_weights_by_name = {n: [1.0] * len(s) for n, s in states_by_name.items()}
    objective_labels_by_name = {n: [n] * len(s) for n, s in states_by_name.items()}

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(base_root, device=str(device), local_files_only=True)
    policy.eval()
    policy.requires_grad_(False)
    modules = dict(policy.named_modules())
    core = policy.model
    expert_model = core.paligemma_with_expert.gemma_expert.model
    language_model = core.paligemma_with_expert.paligemma.model.language_model
    expert_model.config._attn_implementation = "eager"
    language_model.config._attn_implementation = "eager"

    groups = module_groups()
    if backend is not None:
        groups = backend.order_groups(groups)
    targets = target_keys()
    specs_by_group = {g: {i: [] for i in blocks} for g, blocks in groups.items()}
    dense_sources = {}
    with ExitStack() as stack:
        base = stack.enter_context(
            safe_open(base_root / "model.safetensors", framework="pt", device="cpu")
        )
        handles = {
            n: stack.enter_context(safe_open(p / "model.safetensors", framework="pt", device="cpu"))
            for n, p in experts.items()
        }
        prior = (
            stack.enter_context(
                safe_open(prior_root / "model.safetensors", framework="pt", device="cpu")
            )
            if prior_root
            else None
        )
        keys = set(base.keys())
        if (
            not targets <= keys
            or any(set(h.keys()) != keys for h in handles.values())
            or (prior is not None and set(prior.keys()) != keys)
        ):
            raise ValueError(
                "Base, expert and prior tensor schemas differ from supported pi0.5 scope"
            )
        # This release fits the full adapted scope used by the paper. Reject unseen
        # adaptations rather than silently reverting their weights to the base.
        for key in keys:
            base_tensor = base.get_tensor(key)
            for name, handle in handles.items():
                value = handle.get_tensor(key)
                if value.shape != base_tensor.shape or value.dtype != base_tensor.dtype:
                    raise ValueError(f"{name}: incompatible tensor {key}")
                if key not in targets and not torch.equal(value, base_tensor):
                    raise ValueError(f"{name}: adapted tensor outside supported scope: {key}")
            if (
                prior is not None
                and key not in targets
                and not torch.equal(prior.get_tensor(key), base_tensor)
            ):
                raise ValueError(f"Prior changed an out-of-scope tensor: {key}")
        output_tensors = {key: (prior or base).get_tensor(key) for key in base.keys()}
        for group, blocks in groups.items():
            for index, module_names in blocks.items():
                for name in module_names:
                    key = name + ".weight"
                    weights = {n: h.get_tensor(key).float() for n, h in handles.items()}
                    center = (
                        output_tensors[key].float()
                        if prior is not None
                        else sum(weights.values()) / len(names)
                    )
                    output_tensors[key] = center.to(output_tensors[key].dtype).contiguous()
                    # The paper starts from an exported dense mean: its storage
                    # rounding is part of the fixed prior, not deferred to export.
                    center = output_tensors[key].float()
                    module = resolve_module(modules, name)
                    module.weight.data.copy_(center.to(device=device, dtype=module.weight.dtype))
                    specs_by_group[group][index].append(
                        LinearSpec(name, name, key, module, weights, center)
                    )
        for short_name in (*FRONTEND_MODULES, OUTPUT_MODULE):
            name = "model." + short_name
            weights = {n: h.get_tensor(name + ".weight").float() for n, h in handles.items()}
            biases = {n: h.get_tensor(name + ".bias").float() for n, h in handles.items()}
            pw = (
                output_tensors[name + ".weight"].float()
                if prior is not None
                else sum(weights.values()) / len(names)
            )
            pb = (
                output_tensors[name + ".bias"].float()
                if prior is not None
                else sum(biases.values()) / len(names)
            )
            pw = pw.to(output_tensors[name + ".weight"].dtype).float()
            pb = pb.to(output_tensors[name + ".bias"].dtype).float()
            module = resolve_module(modules, name)
            for suffix, value in (("weight", pw), ("bias", pb)):
                parameter = getattr(module, suffix)
                parameter.data.copy_(value.to(device=device, dtype=parameter.dtype))
                output_tensors[name + "." + suffix] = value.to(
                    output_tensors[name + "." + suffix].dtype
                ).contiguous()
            dense_sources[short_name] = (name, weights, biases, pw, pb)
    vision_specs = specs_by_group["vision"]
    language_specs = specs_by_group["language"]
    action_specs = specs_by_group["action"]
    metrics = {}

    def assign_source(specs: list[LinearSpec], name: str) -> None:
        for spec in specs:
            spec.module.weight.data.copy_(
                spec.weights[name].to(device=device, dtype=spec.module.weight.dtype)
            )

    def restore_merged(specs: list[LinearSpec]) -> None:
        for spec in specs:
            spec.module.weight.data.copy_(
                output_tensors[spec.weight_key].to(device=device, dtype=spec.module.weight.dtype)
            )

    def interface_forward(short_name: str, value: torch.Tensor, name: str) -> torch.Tensor:
        module = getattr(core, short_name)
        if args.replay_prefix == "merged":
            return linear_forward(module, value)
        _, expert_weights, expert_biases, _, _ = dense_sources[short_name]
        return F.linear(
            value.to(dtype=module.weight.dtype),
            expert_weights[name].to(device=device, dtype=module.weight.dtype),
            expert_biases[name].to(device=device, dtype=module.bias.dtype),
        )

    def capture_inputs(
        specs: list[LinearSpec],
        name: str,
        forwards: list[Any],
        labels: list[str],
        sample_weights: list[float],
    ) -> dict[str, dict[str, torch.Tensor]]:
        if len(forwards) != len(labels) or len(forwards) != len(sample_weights):
            raise ValueError(
                f"{name}: forward/objective-label/source-weight count mismatch: {len(forwards)} != {len(labels)} != {len(sample_weights)}"
            )
        if backend is not None and backend.student_inputs:
            restore_merged(specs)
        else:
            assign_source(specs, name)
        captures: dict[str, dict[str, list[torch.Tensor]]] = {spec.base_name: {} for spec in specs}
        current_label: str | None = None
        current_sqrt_weight = 1.0
        current_forward_index = 0
        handles = []
        for spec in specs:

            def hook(
                _module: torch.nn.Module, inputs: tuple[Any, ...], module_name: str = spec.base_name
            ) -> None:
                if current_label is None:
                    raise RuntimeError("Linear hook fired without an objective-group label")
                captures[module_name].setdefault(current_label, []).append(
                    selected_rows(inputs[0], name, current_forward_index, module_name)
                    * current_sqrt_weight
                )

            handles.append(spec.module.register_forward_pre_hook(hook))
        try:
            for current_forward_index, (forward, label, sample_weight) in enumerate(
                zip(forwards, labels, sample_weights, strict=True)
            ):
                current_label = label
                current_sqrt_weight = math.sqrt(sample_weight)
                forward()
        finally:
            for handle in handles:
                handle.remove()
        return {
            module_name: {label: torch.cat(values, dim=0) for label, values in grouped.items()}
            for module_name, grouped in captures.items()
        }

    def solve_spec(
        spec: LinearSpec, inputs_by_expert: dict[str, dict[str, dict[str, torch.Tensor]]]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        grouped_inputs: dict[str, torch.Tensor] = {}
        grouped_weights: dict[str, torch.Tensor] = {}
        for name in names:
            for label, value in inputs_by_expert[name][spec.base_name].items():
                if label in grouped_inputs:
                    raise ValueError(f"Duplicate objective-group label: {label}")
                grouped_inputs[label] = value
                grouped_weights[label] = spec.weights[name]
        if backend is not None:
            return backend.solve(
                spec.base_name, grouped_inputs, grouped_weights, spec.prior.to(device)
            )
        return solve_weight_multi(
            grouped_inputs,
            grouped_weights,
            spec.prior.to(device),
            args.ridge_ratio,
            args.ridge_scale,
            args.max_correction_ratio,
            args.expert_loss_normalization,
            full_ridges.get(spec.base_name),
        )

    def solve_dense_module(
        short_name: str, x_by_expert: dict[str, dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _adapter_name, expert_weights, expert_biases, prior_weight, prior_bias = dense_sources[
            short_name
        ]
        grouped_inputs: dict[str, torch.Tensor] = {}
        grouped_weights: dict[str, torch.Tensor] = {}
        for name in names:
            for label, value in x_by_expert[name].items():
                if label in grouped_inputs:
                    raise ValueError(f"Duplicate objective-group label: {label}")
                grouped_inputs[label] = with_ones(value)
                grouped_weights[label] = augmented(expert_weights[name], expert_biases[name])
        if backend is not None:
            return backend.solve(
                f"model.{short_name}",
                grouped_inputs,
                grouped_weights,
                augmented(prior_weight, prior_bias).to(device),
                bias=True,
            )
        return solve_weight_multi(
            grouped_inputs,
            grouped_weights,
            augmented(prior_weight, prior_bias).to(device),
            args.ridge_ratio,
            args.ridge_scale,
            args.max_correction_ratio,
            args.expert_loss_normalization,
            full_ridges.get(f"model.{short_name}"),
        )

    all_states = [state for states in states_by_name.values() for state in states]

    def selected_rows(value, name, index, module_name=""):
        if args.pass_index == 1:
            return sample_rows(value, args.rows_per_call)
        is_vision = ".vision_tower." in module_name
        camera = index % 3 if is_vision else 0
        state_index = index // 3 if is_vision else index
        sample = manifests_by_name[name][0]["samples"][state_index]
        flat = value.reshape(-1, value.shape[-1]).float()
        ids = row_indices(
            len(flat),
            args.rows_per_request,
            sample["flow_index"],
            sample["selected_request_slot"],
            cameras=3 if is_vision else 1,
            camera=camera,
            last_only=args.final_call_only,
        )
        return flat[ids]

    def grouped_state_rows(name: str, value_fn: Any) -> dict[str, torch.Tensor]:
        values = [
            selected_rows(value_fn(state), name, index) * math.sqrt(sample_weight)
            for index, (state, sample_weight) in enumerate(
                zip(states_by_name[name], state_weights_by_name[name], strict=True)
            )
        ]
        return concatenate_grouped(values, objective_labels_by_name[name])

    vision_reference_error: float | None = None
    prefix_kv_reference_error: float | None = None
    expert_prefix_vision_error: float | None = None
    expert_prefix_kv_error: float | None = None
    expert_prefix_kv_errors: list[float] = []
    if any(
        (
            state.vision is None
            or state.language_hidden_reference is None
            or state.language_attention_mask is None
            or (state.language_position_ids is None)
            for state in all_states
        )
    ):
        raise ValueError("Full-prefix calibration is missing vision/language replay tensors")
    vision_transformer = core.paligemma_with_expert.paligemma.model.vision_tower.vision_model
    vision_layers = vision_transformer.encoder.layers
    projector = core.paligemma_with_expert.paligemma.model.multi_modal_projector
    vision_errors: list[float] = []
    for state in all_states:
        assert state.vision is not None and state.language_hidden_reference is not None
        projected = []
        for vision_state in state.vision:
            value = vision_state.hidden
            for layer in vision_layers:
                value = layer(value, vision_state.attention_mask)
            projected.append(projector(vision_transformer.post_layernorm(value)))
        visual = torch.cat(projected, dim=1)
        vision_errors.append(
            float(
                (visual.float() - state.language_hidden_reference[:, : visual.shape[1]].float())
                .abs()
                .max()
            )
        )
    vision_reference_error = max(vision_errors)
    print(
        f"Stored-reference Vision mismatch before sequential replay (calibration-policy dependent): {vision_reference_error:.6g}",
        flush=True,
    )
    kv_errors: list[float] = []
    for state in all_states:
        assert state.language_hidden_reference is not None
        assert state.language_attention_mask is not None
        assert state.language_position_ids is not None
        value = state.language_hidden_reference
        for block_index, layer in enumerate(language_model.layers):
            key, cached_value = decoder_prefix_kv(
                layer, language_model.rotary_emb, value, state.language_position_ids, None
            )
            kv_errors.extend(
                (
                    float((key.float() - state.prefix_keys[block_index].float()).abs().max()),
                    float(
                        (cached_value.float() - state.prefix_values[block_index].float())
                        .abs()
                        .max()
                    ),
                )
            )
            value = manual_action_block(
                layer,
                language_model.rotary_emb,
                value,
                state.language_attention_mask,
                state.language_position_ids,
                None,
                None,
                None,
            )
    prefix_kv_reference_error = max(kv_errors)
    print(
        f"Stored-reference Language-KV mismatch before sequential replay (calibration-policy dependent): {prefix_kv_reference_error:.6g}",
        flush=True,
    )

    def vision_items(states: list[ReplayState]) -> list[Any]:
        return [item for state in states for item in state.vision]

    def vision_objective_labels(name: str) -> list[str]:
        return [
            label
            for state, label in zip(
                states_by_name[name], objective_labels_by_name[name], strict=True
            )
            for _item in state.vision
        ]

    def vision_source_weights(name: str) -> list[float]:
        return [
            sample_weight
            for state, sample_weight in zip(
                states_by_name[name], state_weights_by_name[name], strict=True
            )
            for _item in state.vision
        ]

    for block_index in range(27):
        specs = vision_specs[block_index]
        layer = vision_layers[block_index]
        inputs_by_expert = {
            name: capture_inputs(
                specs,
                name,
                [
                    lambda item=item: layer(item.hidden, item.attention_mask)
                    for item in vision_items(states_by_name[name])
                ],
                vision_objective_labels(name),
                vision_source_weights(name),
            )
            for name in names
        }
        for spec in specs:
            if backend is not None and backend.student_inputs:
                inputs_by_expert = {
                    name: capture_inputs(
                        specs,
                        name,
                        [
                            lambda item=item: layer(item.hidden, item.attention_mask)
                            for item in vision_items(states_by_name[name])
                        ],
                        vision_objective_labels(name),
                        vision_source_weights(name),
                    )
                    for name in names
                }
            merged, module_metrics = solve_spec(spec, inputs_by_expert)
            spec.module.weight.data.copy_(merged.to(device=device, dtype=spec.module.weight.dtype))
            output_tensors[spec.weight_key] = merged.to(
                output_tensors[spec.weight_key].dtype
            ).contiguous()
            metrics[spec.base_name] = {
                "stage": "vision_block_sequential",
                "block_index": block_index,
                "kind": "multi_lora_materialized_dense",
                **module_metrics,
            }
        del inputs_by_expert

        def advance_vision(state: ReplayState) -> None:
            for item in state.vision:
                item.hidden = layer(item.hidden, item.attention_mask)

        advance_by_prefix(
            states_by_name,
            args.replay_prefix,
            lambda name: assign_source(specs, name),
            lambda: restore_merged(specs),
            advance_vision,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        verb = "Merged"
        print(f"{verb} Vision block {block_index:02d}/26", flush=True)
    language_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
    if args.replay_prefix == "expert":
        errors = []
        for state in all_states:
            assert state.vision is not None and state.language_hidden_reference is not None
            projected = [
                projector(vision_transformer.post_layernorm(v.hidden)) for v in state.vision
            ]
            visual = torch.cat(projected, dim=1)
            errors.append(
                float(
                    (visual.float() - state.language_hidden_reference[:, : visual.shape[1]].float())
                    .abs()
                    .max()
                )
            )
        expert_prefix_vision_error = max(errors)
        print(
            f"Expert-prefix Vision-to-language stored-reference max error: {expert_prefix_vision_error:.6g}",
            flush=True,
        )
    for state in all_states:
        assert state.vision is not None and state.language_hidden_reference is not None
        projected = [projector(vision_transformer.post_layernorm(v.hidden)) for v in state.vision]
        visual = torch.cat(projected, dim=1)
        tail = state.language_hidden_reference[:, visual.shape[1] :]
        state.language_hidden = torch.cat((visual.to(tail.dtype), tail), dim=1).to(language_dtype)
    for block_index in range(18):
        specs = language_specs[block_index]
        layer = language_model.layers[block_index]

        def language_forwards(states: list[ReplayState]) -> list[Any]:
            forwards = []
            for state in states:
                assert state.language_hidden is not None
                assert state.language_attention_mask is not None
                assert state.language_position_ids is not None
                forwards.append(
                    lambda state=state: manual_action_block(
                        layer,
                        language_model.rotary_emb,
                        state.language_hidden,
                        state.language_attention_mask,
                        state.language_position_ids,
                        None,
                        None,
                        None,
                    )
                )
            return forwards

        inputs_by_expert = {
            name: capture_inputs(
                specs,
                name,
                language_forwards(states_by_name[name]),
                objective_labels_by_name[name],
                state_weights_by_name[name],
            )
            for name in names
        }
        for spec in specs:
            if backend is not None and backend.student_inputs:
                inputs_by_expert = {
                    name: capture_inputs(
                        specs,
                        name,
                        language_forwards(states_by_name[name]),
                        objective_labels_by_name[name],
                        state_weights_by_name[name],
                    )
                    for name in names
                }
            merged, module_metrics = solve_spec(spec, inputs_by_expert)
            spec.module.weight.data.copy_(merged.to(device=device, dtype=spec.module.weight.dtype))
            output_tensors[spec.weight_key] = merged.to(
                output_tensors[spec.weight_key].dtype
            ).contiguous()
            metrics[spec.base_name] = {
                "stage": "language_block_sequential",
                "block_index": block_index,
                "kind": "multi_lora_materialized_dense",
                **module_metrics,
            }
        del inputs_by_expert

        def advance_language(state: ReplayState) -> None:
            assert state.language_hidden is not None
            assert state.language_attention_mask is not None
            assert state.language_position_ids is not None
            key, value = decoder_prefix_kv(
                layer,
                language_model.rotary_emb,
                state.language_hidden,
                state.language_position_ids,
                None,
            )
            if args.replay_prefix == "expert":
                expert_prefix_kv_errors.extend(
                    (
                        float((key.float() - state.prefix_keys[block_index].float()).abs().max()),
                        float(
                            (value.float() - state.prefix_values[block_index].float()).abs().max()
                        ),
                    )
                )
            state.prefix_keys[block_index] = key
            state.prefix_values[block_index] = value
            state.language_hidden = manual_action_block(
                layer,
                language_model.rotary_emb,
                state.language_hidden,
                state.language_attention_mask,
                state.language_position_ids,
                None,
                None,
                None,
            )

        advance_by_prefix(
            states_by_name,
            args.replay_prefix,
            lambda name: assign_source(specs, name),
            lambda: restore_merged(specs),
            advance_language,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        verb = "Merged"
        print(f"{verb} Language block {block_index:02d}/17", flush=True)
    if args.replay_prefix == "expert":
        expert_prefix_kv_error = max(expert_prefix_kv_errors)
        print(
            f"Expert-prefix Language-KV stored-reference max error: {expert_prefix_kv_error:.6g}",
            flush=True,
        )
    reference_errors: list[float] = []
    for state in all_states:
        action = linear_forward(core.action_in_proj, state.action_input)
        time = F.silu(linear_forward(core.time_mlp_in, state.time_input))
        cond = F.silu(linear_forward(core.time_mlp_out, time))
        reference_errors.extend(
            (
                float((action.float() - state.hidden_reference.float()).abs().max()),
                float((cond.float() - state.cond_reference.float()).abs().max()),
            )
        )
    print(
        f"Stored-reference action front-end mismatch before interface solve (calibration-policy dependent): {max(reference_errors):.6g}",
        flush=True,
    )
    expert_prefix_frontend_error: float | None = None
    if args.replay_prefix == "expert":
        errors = []
        for name in names:
            for state in states_by_name[name]:
                action = interface_forward("action_in_proj", state.action_input, name)
                time = F.silu(interface_forward("time_mlp_in", state.time_input, name))
                cond = F.silu(interface_forward("time_mlp_out", time, name))
                errors.extend(
                    (
                        float((action.float() - state.hidden_reference.float()).abs().max()),
                        float((cond.float() - state.cond_reference.float()).abs().max()),
                    )
                )
        expert_prefix_frontend_error = max(errors)
        print(
            f"Expert-prefix action front-end stored-reference max error: {expert_prefix_frontend_error:.6g}",
            flush=True,
        )
    from transformers.cache_utils import DynamicCache

    validation_state = all_states[0]
    validation_cache = DynamicCache(
        tuple(
            (
                (keys.clone(), values.clone(), None)
                for keys, values in zip(
                    validation_state.prefix_keys, validation_state.prefix_values, strict=True
                )
            )
        )
    )
    validation_position_embeddings = expert_model.rotary_emb(
        validation_state.hidden_reference, validation_state.position_ids
    )
    validation_cache_position = torch.arange(
        validation_state.prefix_keys[0].shape[-2],
        validation_state.prefix_keys[0].shape[-2] + validation_state.hidden_reference.shape[-2],
        device=device,
    )
    native_block_output = expert_model.layers[0](
        validation_state.hidden_reference,
        attention_mask=validation_state.attention_mask,
        position_ids=validation_state.position_ids,
        past_key_values=validation_cache,
        use_cache=False,
        cache_position=validation_cache_position,
        position_embeddings=validation_position_embeddings,
        adarms_cond=validation_state.cond_reference,
    )
    manual_block_output = manual_action_block(
        expert_model.layers[0],
        expert_model.rotary_emb,
        validation_state.hidden_reference,
        validation_state.attention_mask,
        validation_state.position_ids,
        validation_state.prefix_keys[0],
        validation_state.prefix_values[0],
        validation_state.cond_reference,
    )
    manual_replay_error = float(
        (native_block_output.float() - manual_block_output.float()).abs().max()
    )
    if manual_replay_error > 0.02:
        raise RuntimeError(
            f"Manual Action block replay disagrees with native layer: {manual_replay_error}"
        )
    print(f"Manual/native Action block max error: {manual_replay_error:.6g}", flush=True)
    action_solution, action_metrics = solve_dense_module(
        "action_in_proj",
        {name: grouped_state_rows(name, lambda state: state.action_input) for name in names},
    )
    time_in_solution, time_in_metrics = solve_dense_module(
        "time_mlp_in",
        {name: grouped_state_rows(name, lambda state: state.time_input) for name in names},
    )
    for short_name, solution, module_metrics in (
        ("action_in_proj", action_solution, action_metrics),
        ("time_mlp_in", time_in_solution, time_in_metrics),
    ):
        module = getattr(core, short_name)
        module.weight.data.copy_(solution[:, :-1].to(device=device, dtype=module.weight.dtype))
        module.bias.data.copy_(solution[:, -1].to(device=device, dtype=module.bias.dtype))
        base_name = f"model.{short_name}"
        output_tensors[f"{base_name}.weight"] = solution[:, :-1].to(
            output_tensors[f"{base_name}.weight"].dtype
        )
        output_tensors[f"{base_name}.bias"] = solution[:, -1].to(
            output_tensors[f"{base_name}.bias"].dtype
        )
        metrics[base_name] = {
            "stage": "front_end",
            "kind": "multi_dense_augmented",
            **module_metrics,
        }
    time_hidden_by_expert = {
        name: grouped_state_rows(
            name,
            lambda state, name=name: F.silu(
                interface_forward("time_mlp_in", state.time_input, name)
            ),
        )
        for name in names
    }
    time_out_solution, time_out_metrics = solve_dense_module("time_mlp_out", time_hidden_by_expert)
    core.time_mlp_out.weight.data.copy_(
        time_out_solution[:, :-1].to(device=device, dtype=core.time_mlp_out.weight.dtype)
    )
    core.time_mlp_out.bias.data.copy_(
        time_out_solution[:, -1].to(device=device, dtype=core.time_mlp_out.bias.dtype)
    )
    output_tensors["model.time_mlp_out.weight"] = time_out_solution[:, :-1].to(
        output_tensors["model.time_mlp_out.weight"].dtype
    )
    output_tensors["model.time_mlp_out.bias"] = time_out_solution[:, -1].to(
        output_tensors["model.time_mlp_out.bias"].dtype
    )
    metrics["model.time_mlp_out"] = {
        "stage": "front_end_sequential",
        "kind": "multi_dense_augmented",
        **time_out_metrics,
    }
    first_weight_dtype = expert_model.layers[0].self_attn.q_proj.weight.dtype
    for name in names:
        for state in states_by_name[name]:
            state.hidden = interface_forward("action_in_proj", state.action_input, name).to(
                first_weight_dtype
            )
            time = F.silu(interface_forward("time_mlp_in", state.time_input, name))
            state.cond = F.silu(interface_forward("time_mlp_out", time, name))

    def run_and_capture(
        block_index: int, states: list[ReplayState], specs: list[LinearSpec], name: str
    ) -> dict[str, dict[str, torch.Tensor]]:
        forwards = []
        for state in states:
            assert state.hidden is not None and state.cond is not None
            forwards.append(
                lambda state=state: manual_action_block(
                    expert_model.layers[block_index],
                    expert_model.rotary_emb,
                    state.hidden,
                    state.attention_mask,
                    state.position_ids,
                    state.prefix_keys[block_index],
                    state.prefix_values[block_index],
                    state.cond,
                )
            )
        return capture_inputs(
            specs, name, forwards, objective_labels_by_name[name], state_weights_by_name[name]
        )

    for block_index in range(18):
        specs = action_specs[block_index]
        inputs_by_expert = {
            name: run_and_capture(block_index, states_by_name[name], specs, name) for name in names
        }
        for spec in specs:
            if backend is not None and backend.student_inputs:
                inputs_by_expert = {
                    name: run_and_capture(block_index, states_by_name[name], specs, name)
                    for name in names
                }
            merged, module_metrics = solve_spec(spec, inputs_by_expert)
            if not torch.isfinite(merged).all():
                raise ValueError(f"Non-finite solution for {spec.base_name}")
            spec.module.weight.data.copy_(merged.to(device=device, dtype=spec.module.weight.dtype))
            output_tensors[spec.weight_key] = merged.to(
                output_tensors[spec.weight_key].dtype
            ).contiguous()
            metrics[spec.base_name] = {
                "stage": "action_block_sequential",
                "block_index": block_index,
                "kind": "multi_lora_materialized_dense",
                **module_metrics,
            }
        del inputs_by_expert

        def advance_action(state: ReplayState) -> None:
            assert state.hidden is not None and state.cond is not None
            state.hidden = manual_action_block(
                expert_model.layers[block_index],
                expert_model.rotary_emb,
                state.hidden,
                state.attention_mask,
                state.position_ids,
                state.prefix_keys[block_index],
                state.prefix_values[block_index],
                state.cond,
            )

        advance_by_prefix(
            states_by_name,
            args.replay_prefix,
            lambda name: assign_source(specs, name),
            lambda: restore_merged(specs),
            advance_action,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        block_improvements = [metrics[spec.base_name]["improvement_vs_prior"] for spec in specs]
        print(
            f"Merged Action block {block_index:02d}/17: mean offline improvement={mean(block_improvements):.2%}",
            flush=True,
        )

    def final_hidden(state: ReplayState) -> torch.Tensor:
        assert state.hidden is not None and state.cond is not None
        value, _ = expert_model.norm(state.hidden, state.cond)
        return value[:, -core.config.chunk_size :].float()

    output_solution, output_metrics = solve_dense_module(
        OUTPUT_MODULE, {name: grouped_state_rows(name, final_hidden) for name in names}
    )
    output_tensors["model.action_out_proj.weight"] = output_solution[:, :-1].to(
        output_tensors["model.action_out_proj.weight"].dtype
    )
    output_tensors["model.action_out_proj.bias"] = output_solution[:, -1].to(
        output_tensors["model.action_out_proj.bias"].dtype
    )
    metrics["model.action_out_proj"] = {
        "stage": "output_head_sequential",
        "kind": "multi_dense_augmented",
        **output_metrics,
    }
    realized_rows = validate_rows(metrics, names, pass_index, cfg, require_ridge=backend is None)
    if backend is not None and backend.teacher_only:
        return backend.finish_teacher(expert_hashes, cache_hashes, manual_replay_error)
    if any(not torch.isfinite(t).all() for t in output_tensors.values()):
        raise ValueError("Merged checkpoint contains NaN or Inf")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tcr-staging-", dir=output.parent))
    copy_support(experts[names[0]], staging, output)
    save_file(
        {k: v.contiguous() for k, v in output_tensors.items()},
        staging / "model.safetensors",
        metadata={"format": "pt"},
    )
    manifest = dict(
        schema_version=1,
        method="TCR" if backend is None else backend.method,
        pass_index=pass_index,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=cfg,
        expert_sha256=expert_hashes,
        cache_sha256=cache_hashes,
        cache_manifest_sha256={
            n: sha256(Path(e[cache_key]) / "replay.json") for n, e in cfg["experts"].items()
        },
        base_model_sha256=sha256(base_root / "model.safetensors"),
        prior_model_sha256=sha256(prior_root / "model.safetensors") if prior_root else None,
        ridge_reference=str(ridge_root) if ridge_root else None,
        model_sha256=sha256(staging / "model.safetensors"),
        module_count=len(metrics),
        modified_tensor_count=len(targets),
        realized_row_total=realized_rows,
        modules=metrics,
        manual_native_block_max_error=manual_replay_error,
        diagnostic_note="Stored-reference discrepancies are not native replay errors when weights differ.",
    )
    if backend is not None:
        manifest["baseline_settings"] = backend.settings
        manifest["recipe"] = backend.description
    manifest_name = "tcr_manifest.json" if backend is None else "baseline_manifest.json"
    (staging / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(output)
    print(
        json.dumps(
            {"output": str(output), "module_count": len(metrics), "realized_rows": realized_rows}
        )
    )
    return manifest
