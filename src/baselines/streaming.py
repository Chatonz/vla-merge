"""Memory-bounded exact global task-vector fusion for billion-scale domains."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import torch

from .parameter_fusion import ParameterFusionError


def _validate_vectors(vectors: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], int]:
    if not vectors:
        raise ParameterFusionError("at least one task vector is required")
    values = [value.detach().cpu() for value in vectors]
    shape = values[0].shape
    if len(shape) != 1 or any(value.shape != shape for value in values):
        raise ParameterFusionError("task vectors must be equally shaped flat tensors")
    if any(not value.is_floating_point() for value in values):
        raise ParameterFusionError("task vectors must be floating point")
    return values, values[0].numel()


def _chunks(length: int, chunk_size: int):
    if chunk_size <= 0:
        raise ParameterFusionError("chunk_size must be positive")
    for start in range(0, length, chunk_size):
        yield start, min(start + chunk_size, length)


def exact_global_keep_threshold_chunked(
    delta: torch.Tensor,
    keep_fraction: float,
    *,
    chunk_size: int = 4 * 1024 * 1024,
) -> float | None:
    """Exact historical TIES threshold without constructing ``abs(delta)`` globally.

    Nonnegative finite IEEE float32 values have the same ordering as their
    unsigned bit patterns. Four radix-counting passes therefore recover the
    exact kth value and avoid library/index limits above 2^31 coordinates.
    """
    values, length = _validate_vectors([delta])
    value = values[0]
    if not 0.0 < keep_fraction <= 1.0:
        raise ParameterFusionError("keep_fraction must lie in (0, 1]")
    retained = int(length * keep_fraction)
    if retained < 1:
        raise ParameterFusionError("keep_fraction retains no coordinates")
    if retained == length:
        return None
    rank = length - retained  # one-based, matching exact_global_keep_threshold
    prefix = 0
    for shift in (24, 16, 8, 0):
        counts = torch.zeros(256, dtype=torch.int64)
        prefix_shift = shift + 8
        for start, end in _chunks(length, chunk_size):
            chunk = value[start:end].float()
            if not torch.isfinite(chunk).all():
                raise ParameterFusionError("task vector is non-finite")
            bits = chunk.abs().contiguous().view(torch.int32).to(torch.int64)
            if prefix_shift < 32:
                bits = bits[(bits >> prefix_shift) == prefix]
            bucket = (bits >> shift) & 0xFF
            counts.add_(torch.bincount(bucket, minlength=256))
        cumulative = torch.cumsum(counts, dim=0)
        candidates = torch.nonzero(cumulative >= rank, as_tuple=False)
        if candidates.numel() == 0:
            raise RuntimeError("radix selection lost the requested rank")
        selected = int(candidates[0, 0])
        before = int(cumulative[selected - 1]) if selected else 0
        rank -= before
        prefix = (prefix << 8) | selected
    return struct.unpack("!f", struct.pack("!I", prefix))[0]


def _trimmed(chunk: torch.Tensor, threshold: float | None) -> torch.Tensor:
    # The repository's audited reference kernel promotes task vectors to
    # float64. Keep that numerical definition while bounding temporary memory.
    value = chunk.double()
    return value if threshold is None else value * (value.abs() >= threshold)


def ties_merge_flat_chunked(
    task_vectors: Sequence[torch.Tensor],
    *,
    keep_fraction: float,
    alpha: float,
    disjoint: str = "mean",
    chunk_size: int = 4 * 1024 * 1024,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Exact global TIES over a flat domain with bounded temporary memory."""
    values, length = _validate_vectors(task_vectors)
    if disjoint not in {"mean", "sum"}:
        raise ParameterFusionError("disjoint must be mean or sum")
    if not math.isfinite(alpha):
        raise ParameterFusionError("alpha must be finite")
    thresholds = [
        exact_global_keep_threshold_chunked(value, keep_fraction, chunk_size=chunk_size)
        for value in values
    ]
    retained_counts = [0] * len(values)
    balance = 0
    conflict_coordinates = 0 if len(values) == 2 else None
    for start, end in _chunks(length, chunk_size):
        trimmed = [
            _trimmed(value[start:end], threshold)
            for value, threshold in zip(values, thresholds, strict=True)
        ]
        for index, value in enumerate(trimmed):
            retained_counts[index] += int(torch.count_nonzero(value))
        elected = torch.sign(torch.stack(trimmed).sum(dim=0))
        balance += int(elected.to(torch.int64).sum())
        if conflict_coordinates is not None:
            conflict_coordinates += int(
                torch.count_nonzero(
                    (trimmed[0] != 0)
                    & (trimmed[1] != 0)
                    & (torch.sign(trimmed[0]) != torch.sign(trimmed[1]))
                )
            )
    zero_sign = 1 if balance > 0 else -1 if balance < 0 else 0
    merged = torch.empty(length, dtype=torch.float64)
    merged_nonzero = 0
    for start, end in _chunks(length, chunk_size):
        trimmed = [
            _trimmed(value[start:end], threshold)
            for value, threshold in zip(values, thresholds, strict=True)
        ]
        elected = torch.sign(torch.stack(trimmed).sum(dim=0))
        if zero_sign:
            elected = torch.where(elected == 0, float(zero_sign), elected)
        selected = [value * torch.where(elected > 0, value > 0, value < 0) for value in trimmed]
        combined = torch.stack(selected).sum(dim=0)
        if disjoint == "mean":
            contributors = torch.stack([value != 0 for value in selected]).sum(dim=0)
            combined.div_(contributors.clamp_min(1))
        combined.mul_(float(alpha))
        merged[start:end] = combined
        merged_nonzero += int(torch.count_nonzero(combined))
    return merged, {
        "thresholds": thresholds,
        "retained_counts": retained_counts,
        "merged_nonzero": merged_nonzero,
        "conflict_coordinates": conflict_coordinates,
        "zero_sign_fallback": zero_sign,
        "disjoint": disjoint,
        "keep_fraction": keep_fraction,
        "alpha": alpha,
        "selection": "exact_ieee_float32_radix_global",
        "chunk_size": chunk_size,
    }


def dare_ties_merge_flat_chunked(
    task_vectors: Sequence[torch.Tensor],
    *,
    drop_probability: float,
    seed: int,
    alpha: float,
    disjoint: str = "mean",
    chunk_size: int = 8 * 1024 * 1024,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Deterministic DARE rescaling and global sign election, without a second trim."""
    values, length = _validate_vectors(task_vectors)
    if not 0.0 <= drop_probability < 1.0:
        raise ParameterFusionError("drop_probability must lie in [0, 1)")
    if disjoint not in {"mean", "sum"} or not math.isfinite(alpha):
        raise ParameterFusionError("invalid disjoint or alpha")
    keep_probability = 1.0 - drop_probability
    seeds = [int(seed) + index for index in range(len(values))]

    def generators():
        result = []
        for task_seed in seeds:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(task_seed)
            result.append(generator)
        return result

    retained_counts = [0] * len(values)
    balance = 0
    first_generators = generators()
    for start, end in _chunks(length, chunk_size):
        sparse = []
        for index, (value, generator) in enumerate(zip(values, first_generators, strict=True)):
            keep = torch.rand(end - start, generator=generator) < keep_probability
            retained_counts[index] += int(keep.sum())
            sparse.append(value[start:end].double() * keep / keep_probability)
        balance += int(torch.sign(torch.stack(sparse).sum(dim=0)).to(torch.int64).sum())
    zero_sign = 1 if balance > 0 else -1 if balance < 0 else 0
    second_generators = generators()
    merged = torch.empty(length, dtype=torch.float64)
    merged_nonzero = 0
    for start, end in _chunks(length, chunk_size):
        sparse = []
        for value, generator in zip(values, second_generators, strict=True):
            keep = torch.rand(end - start, generator=generator) < keep_probability
            sparse.append(value[start:end].double() * keep / keep_probability)
        elected = torch.sign(torch.stack(sparse).sum(dim=0))
        if zero_sign:
            elected = torch.where(elected == 0, float(zero_sign), elected)
        selected = [value * torch.where(elected > 0, value > 0, value < 0) for value in sparse]
        combined = torch.stack(selected).sum(dim=0)
        if disjoint == "mean":
            contributors = torch.stack([value != 0 for value in selected]).sum(dim=0)
            combined.div_(contributors.clamp_min(1))
        combined.mul_(float(alpha))
        merged[start:end] = combined
        merged_nonzero += int(torch.count_nonzero(combined))
    return merged, {
        "thresholds": [None] * len(values),
        "retained_counts": retained_counts,
        "merged_nonzero": merged_nonzero,
        "zero_sign_fallback": zero_sign,
        "disjoint": disjoint,
        "keep_fraction": 1.0,
        "alpha": alpha,
        "dare_drop_probability": drop_probability,
        "dare_keep_probability": keep_probability,
        "dare_seeds": seeds,
        "dare_retained_counts": retained_counts,
        "second_ties_trim": False,
        "chunk_size": chunk_size,
    }
