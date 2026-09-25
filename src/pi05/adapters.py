"""Materialize an exact dense PI0.5 checkpoint from a PEFT LoRA adapter.

This is used when a PEFT candidate is numerically valid for rollout but hooks
must observe the dense module path used by full-weight PI0.5 candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER_PREFIX = "base_model.model."
LORA_A_MARKER = ".lora_A."
LORA_B_MARKER = ".lora_B."


class TensorMetadata(NamedTuple):
    shape: tuple[int, ...]
    dtype: str


class TensorOperation(NamedTuple):
    kind: str
    base_key: str
    adapter_key: str
    pair_key: str | None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_same_source(root: Path, output: Path, names: tuple[str, ...]) -> None:
    for name in names:
        src = root / name
        if src.is_file():
            shutil.copy2(src, output / name)


def compile_include_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"Invalid include pattern {pattern!r}: {exc}") from exc
    return compiled


def read_safetensors_schema(path: Path) -> dict[str, TensorMetadata]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            return {
                key: TensorMetadata(
                    tuple(handle.get_slice(key).get_shape()), str(handle.get_slice(key).get_dtype())
                )
                for key in handle.keys()
            }
    except Exception as exc:
        raise ValueError(f"Invalid safetensors file {path}: {exc}") from exc


def validate_adapter_config(adapter_config: dict) -> tuple[int, float, float]:
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("Only PEFT LoRA adapters are supported")
    try:
        raw_rank = adapter_config["r"]
        rank = int(raw_rank)
        alpha = float(adapter_config["lora_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("LoRA adapter_config requires numeric r and lora_alpha") from exc
    if isinstance(raw_rank, bool) or rank <= 0:
        raise ValueError("Invalid LoRA rank")
    try:
        if float(raw_rank) != rank:
            raise ValueError("LoRA rank must be an integer")
    except (TypeError, ValueError) as exc:
        raise ValueError("LoRA rank must be an integer") from exc
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid LoRA alpha")
    unsupported = {
        "rank_pattern": bool(adapter_config.get("rank_pattern")),
        "alpha_pattern": bool(adapter_config.get("alpha_pattern")),
        "use_dora": bool(adapter_config.get("use_dora")),
        "use_rslora": bool(adapter_config.get("use_rslora")),
        "fan_in_fan_out": bool(adapter_config.get("fan_in_fan_out")),
        "lora_bias": bool(adapter_config.get("lora_bias")),
        "target_parameters": bool(adapter_config.get("target_parameters")),
        "layer_replication": bool(adapter_config.get("layer_replication")),
    }
    enabled = sorted((name for name, value in unsupported.items() if value))
    if adapter_config.get("bias") not in (None, "none"):
        enabled.append("bias")
    if enabled:
        raise ValueError(
            "Unsupported PEFT LoRA features for exact alpha/r materialization: "
            + ", ".join(enabled)
        )
    return (rank, alpha, alpha / rank)


def _paired_key(key: str, marker: str, paired_marker: str) -> str:
    prefix, suffix = key.split(marker, 1)
    return f"{prefix}{paired_marker}{suffix}"


def build_tensor_plan(
    adapter_schema: dict[str, TensorMetadata],
    base_schema: dict[str, TensorMetadata],
    *,
    rank: int,
    include_patterns: list[re.Pattern[str]],
) -> tuple[list[TensorOperation], dict]:
    """Validate adapter/base headers and return the selected dense operations.

    All adapter entries are validated even when an include pattern excludes them.
    This prevents a partial selection from hiding an invalid or incompatible input
    adapter. No tensor payload is loaded by this function.
    """

    def included(*names: str) -> bool:
        return not include_patterns or any(
            (pattern.search(name) for pattern in include_patterns for name in names)
        )

    keys = set(adapter_schema)
    a_keys = sorted((key for key in keys if LORA_A_MARKER in key))
    b_keys = sorted((key for key in keys if LORA_B_MARKER in key))
    direct_keys = sorted(keys - set(a_keys) - set(b_keys))
    if not keys:
        raise ValueError("Adapter safetensors contains no tensors")
    missing_b = [
        key for key in a_keys if _paired_key(key, LORA_A_MARKER, LORA_B_MARKER) not in keys
    ]
    if missing_b:
        raise ValueError(f"Missing LoRA B tensor for {missing_b[0]}")
    missing_a = [
        key for key in b_keys if _paired_key(key, LORA_B_MARKER, LORA_A_MARKER) not in keys
    ]
    if missing_a:
        raise ValueError(f"Missing LoRA A tensor for {missing_a[0]}")
    operations: list[TensorOperation] = []
    schema_rows: list[dict] = []
    mapped_base_keys: dict[str, str] = {}
    selected_lora = 0
    selected_direct = 0
    for key in a_keys:
        pair_key = _paired_key(key, LORA_A_MARKER, LORA_B_MARKER)
        module_name = key.split(LORA_A_MARKER, 1)[0]
        if not module_name.startswith(ADAPTER_PREFIX):
            raise ValueError(f"Cannot map adapter module to base: {module_name}")
        base_module = module_name[len(ADAPTER_PREFIX) :]
        base_key = f"{base_module}.weight"
        if base_key not in base_schema:
            raise ValueError(f"Missing base tensor {base_key} for {module_name}")
        a_meta = adapter_schema[key]
        b_meta = adapter_schema[pair_key]
        base_meta = base_schema[base_key]
        if len(a_meta.shape) != 2 or len(b_meta.shape) != 2 or len(base_meta.shape) != 2:
            raise ValueError(
                f"LoRA and base weight tensors must be 2D for {key} -> {base_key}: A={a_meta.shape}, B={b_meta.shape}, base={base_meta.shape}"
            )
        if a_meta.shape[0] != rank or b_meta.shape[1] != rank:
            raise ValueError(
                f"LoRA rank mismatch for {key}: config={rank}, A={a_meta.shape}, B={b_meta.shape}"
            )
        expected_base_shape = (b_meta.shape[0], a_meta.shape[1])
        if b_meta.shape[1] != a_meta.shape[0] or base_meta.shape != expected_base_shape:
            raise ValueError(
                f"LoRA shape mismatch for {key} -> {base_key}: A={a_meta.shape}, B={b_meta.shape}, base={base_meta.shape}"
            )
        if base_key in mapped_base_keys:
            raise ValueError(
                f"Multiple adapter tensors map to base tensor {base_key}: {mapped_base_keys[base_key]} and {key}"
            )
        mapped_base_keys[base_key] = key
        schema_rows.append(
            {
                "kind": "lora",
                "base_key": base_key,
                "base_shape": list(base_meta.shape),
                "base_dtype": base_meta.dtype,
                "a_shape": list(a_meta.shape),
                "a_dtype": a_meta.dtype,
                "b_shape": list(b_meta.shape),
                "b_dtype": b_meta.dtype,
            }
        )
        if included(base_module, base_key):
            operations.append(TensorOperation("lora", base_key, key, pair_key))
            selected_lora += 1
    for key in direct_keys:
        if not key.startswith(ADAPTER_PREFIX):
            raise ValueError(f"Cannot map direct saved tensor to base: {key}")
        base_key = key[len(ADAPTER_PREFIX) :]
        if base_key not in base_schema:
            raise ValueError(f"Missing base tensor {base_key} for {key}")
        adapter_meta = adapter_schema[key]
        base_meta = base_schema[base_key]
        if adapter_meta.shape != base_meta.shape:
            raise ValueError(
                f"Shape mismatch for {key} -> {base_key}: adapter={adapter_meta.shape}, base={base_meta.shape}"
            )
        if base_key in mapped_base_keys:
            raise ValueError(
                f"Multiple adapter tensors map to base tensor {base_key}: {mapped_base_keys[base_key]} and {key}"
            )
        mapped_base_keys[base_key] = key
        schema_rows.append(
            {
                "kind": "direct",
                "base_key": base_key,
                "base_shape": list(base_meta.shape),
                "base_dtype": base_meta.dtype,
                "adapter_shape": list(adapter_meta.shape),
                "adapter_dtype": adapter_meta.dtype,
            }
        )
        if included(base_key):
            operations.append(TensorOperation("direct", base_key, key, None))
            selected_direct += 1
    if not operations:
        raise ValueError("No adapted tensors matched the include patterns")
    schema_rows.sort(key=lambda row: (row["base_key"], row["kind"]))
    signature_payload = json.dumps(schema_rows, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    summary = {
        "adapter_tensor_count": len(adapter_schema),
        "base_tensor_count": len(base_schema),
        "lora_pair_count_total": len(a_keys),
        "lora_pair_count_included": selected_lora,
        "direct_tensor_count_total": len(direct_keys),
        "direct_tensor_count_included": selected_direct,
        "logical_adapted_tensor_count_total": len(a_keys) + len(direct_keys),
        "modified_tensor_count": len(operations),
        "schema_signature_sha256": hashlib.sha256(signature_payload).hexdigest(),
    }
    return (operations, summary)


def build_materialization_plan(
    adapter_root: Path, base_root: Path, include_pattern_strings: list[str] | None = None
) -> tuple[dict, list[TensorOperation], dict]:
    adapter_root = adapter_root.expanduser().resolve()
    base_root = base_root.expanduser().resolve()
    if not adapter_root.is_dir():
        raise FileNotFoundError(adapter_root)
    if not base_root.is_dir():
        raise FileNotFoundError(base_root)
    adapter_config = load_json(adapter_root / "adapter_config.json")
    rank, alpha, lora_scale = validate_adapter_config(adapter_config)
    adapter_schema = read_safetensors_schema(adapter_root / "adapter_model.safetensors")
    base_schema = read_safetensors_schema(base_root / "model.safetensors")
    include_patterns = compile_include_patterns(include_pattern_strings or [])
    operations, summary = build_tensor_plan(
        adapter_schema, base_schema, rank=rank, include_patterns=include_patterns
    )
    settings = {"rank": rank, "alpha": alpha, "lora_scale": lora_scale}
    return (settings, operations, summary)


def peft_cpu_safe_merge_lora_weight(
    base_value: torch.Tensor, a_value: torch.Tensor, b_value: torch.Tensor, lora_scale: float
) -> torch.Tensor:
    """Reproduce PEFT CPU safe-merge casting and addition order exactly."""
    if base_value.ndim != 2 or a_value.ndim != 2 or b_value.ndim != 2:
        raise ValueError("PEFT-safe LoRA merge requires 2-D tensors")
    if b_value.shape[1] != a_value.shape[0] or (b_value.shape[0], a_value.shape[1]) != tuple(
        base_value.shape
    ):
        raise ValueError("PEFT-safe LoRA merge shapes differ")
    if not math.isfinite(float(lora_scale)):
        raise ValueError("PEFT-safe LoRA scale must be finite")
    base = base_value.detach().cpu()
    delta_fp32 = (b_value.detach().cpu().float() @ a_value.detach().cpu().float()).mul_(
        float(lora_scale)
    )
    delta_in_base_dtype = delta_fp32.to(dtype=base.dtype)
    merged = base + delta_in_base_dtype
    if not torch.isfinite(merged).all():
        raise ValueError("PEFT-safe LoRA merge produced non-finite values")
    return merged.contiguous()


def materialize_checkpoint(
    *,
    adapter_root: Path,
    base_root: Path,
    output: Path,
    include_pattern_strings: list[str] | None = None,
) -> dict:
    adapter_root = adapter_root.expanduser().resolve()
    base_root = base_root.expanduser().resolve()
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    include_pattern_strings = include_pattern_strings or []
    settings, operations, schema_summary = build_materialization_plan(
        adapter_root, base_root, include_pattern_strings
    )
    adapter_weights = adapter_root / "adapter_model.safetensors"
    base_weights = base_root / "model.safetensors"
    with safe_open(base_weights, framework="pt", device="cpu") as base:
        dense_tensors = {key: base.get_tensor(key) for key in base.keys()}
    modified_keys: set[str] = set()
    with safe_open(adapter_weights, framework="pt", device="cpu") as adapter:
        for operation in operations:
            base_value = dense_tensors[operation.base_key]
            if operation.kind == "lora":
                assert operation.pair_key is not None
                a_value = adapter.get_tensor(operation.adapter_key)
                b_value = adapter.get_tensor(operation.pair_key)
                dense_tensors[operation.base_key] = peft_cpu_safe_merge_lora_weight(
                    base_value, a_value, b_value, settings["lora_scale"]
                )
            else:
                dense_tensors[operation.base_key] = adapter.get_tensor(
                    operation.adapter_key
                ).contiguous()
            modified_keys.add(operation.base_key)
    output.mkdir(parents=True)
    copy_if_same_source(
        adapter_root,
        output,
        (
            "README.md",
            "train_config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ),
    )
    if (adapter_root / "tokenizer").is_dir():
        shutil.copytree(adapter_root / "tokenizer", output / "tokenizer")
    if (adapter_root / "config.json").is_file():
        policy_config = load_json(adapter_root / "config.json")
    else:
        policy_config = load_json(base_root / "config.json")
    policy_config["use_peft"] = False
    policy_config["pretrained_path"] = str(output)
    (output / "config.json").write_text(
        json.dumps(policy_config, indent=4, sort_keys=False) + "\n", encoding="utf-8"
    )
    output_weights = output / "model.safetensors"
    save_file(dense_tensors, output_weights, metadata={"format": "pt"})
    manifest = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "pi05_exact_peft_lora_to_dense",
        "training": False,
        "semantic_equivalence": "dense checkpoint equals PEFT CPU merge_and_unload safe-merged state",
        "rounding_semantics": {
            "matmul": "float32_BA",
            "first_round": "cast_delta_to_base_weight_dtype",
            "addition": "base_weight_dtype_add",
            "forbidden_shortcut": "cast_once_after_base_float_plus_delta_float",
        },
        "source_rank": settings["rank"],
        "source_alpha": settings["alpha"],
        "source_peft_scale": settings["lora_scale"],
        "include_patterns": include_pattern_strings,
        "lora_pair_count": schema_summary["lora_pair_count_included"],
        "modules_to_save_tensor_count": schema_summary["direct_tensor_count_included"],
        "modified_tensor_count": len(modified_keys),
        "schema_validation": {"status": "passed", **schema_summary},
        "inputs": {
            "adapter": str(adapter_root),
            "base_model": str(base_root),
            "adapter_sha256": sha256(adapter_weights),
            "adapter_config_sha256": sha256(adapter_root / "adapter_config.json"),
            "base_model_sha256": sha256(base_weights),
        },
        "output": str(output),
        "model_sha256": sha256(output_weights),
    }
    (output / "dense_equivalent_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
