"""Data-free parameter-fusion kernels for full Psi0 action headers."""

from __future__ import annotations

import math
from typing import Sequence

import torch


class ParameterFusionError(ValueError):
    pass


def _fp64_inputs(values: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if not values:
        raise ParameterFusionError("at least one tensor is required")
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ParameterFusionError("tensor shapes differ")
    if any(not value.is_floating_point() for value in values):
        raise ParameterFusionError("fusion inputs must be floating point")
    converted = [value.detach().cpu().double() for value in values]
    if any(not torch.isfinite(value).all() for value in converted):
        raise ParameterFusionError("fusion input is non-finite")
    return converted


def weighted_soup(experts: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    values = _fp64_inputs(experts)
    coefficients = torch.tensor(list(weights), dtype=torch.float64)
    if coefficients.shape != (len(values),):
        raise ParameterFusionError("weight count differs from expert count")
    if not torch.isfinite(coefficients).all() or torch.any(coefficients < 0):
        raise ParameterFusionError("weights must be finite and nonnegative")
    if not torch.isclose(coefficients.sum(), torch.tensor(1.0, dtype=torch.float64)):
        raise ParameterFusionError("weights must sum to one")
    result = torch.zeros_like(values[0])
    for coefficient, value in zip(coefficients, values):
        result.add_(value, alpha=float(coefficient))
    return result


def task_arithmetic(
    base: torch.Tensor,
    experts: Sequence[torch.Tensor],
    *,
    alpha: float,
) -> torch.Tensor:
    values = _fp64_inputs([base, *experts])
    anchor, task_values = values[0], values[1:]
    if not math.isfinite(alpha):
        raise ParameterFusionError("alpha must be finite")
    result = anchor.clone()
    for value in task_values:
        result.add_(value - anchor, alpha=float(alpha))
    return result


def task_vector_slerp(
    base: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    t: float,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, str]:
    """Spherical interpolation of per-tensor task vectors around a shared base."""
    anchor, first, second = _fp64_inputs([base, left, right])
    if not 0.0 <= t <= 1.0 or not math.isfinite(t):
        raise ParameterFusionError("t must be finite and lie in [0, 1]")
    delta_left = first - anchor
    delta_right = second - anchor
    norm_left = torch.linalg.vector_norm(delta_left)
    norm_right = torch.linalg.vector_norm(delta_right)
    if norm_left <= epsilon or norm_right <= epsilon:
        return anchor + (1.0 - t) * delta_left + t * delta_right, "linear_zero_norm"
    cosine = torch.clamp(
        torch.dot(delta_left.reshape(-1), delta_right.reshape(-1)) / (norm_left * norm_right),
        -1.0,
        1.0,
    )
    omega = torch.acos(cosine)
    sine = torch.sin(omega)
    if torch.abs(sine) <= epsilon:
        return anchor + (1.0 - t) * delta_left + t * delta_right, "linear_collinear"
    interpolated = (
        torch.sin((1.0 - t) * omega) / sine * delta_left + torch.sin(t * omega) / sine * delta_right
    )
    return anchor + interpolated, "spherical"


def exact_global_keep_threshold(delta: torch.Tensor, keep_fraction: float) -> float | None:
    """Return the historical one-based TIES threshold over one complete vector."""
    if delta.ndim != 1 or not delta.is_floating_point():
        raise ParameterFusionError("TIES delta must be a floating flat vector")
    if not 0.0 < keep_fraction <= 1.0:
        raise ParameterFusionError("keep_fraction must lie in (0, 1]")
    retained = int(delta.numel() * keep_fraction)
    if retained < 1:
        raise ParameterFusionError("keep_fraction retains no coordinates")
    if retained == delta.numel():
        return None
    kth_one_based = delta.numel() - retained
    return float(torch.kthvalue(delta.abs(), kth_one_based).values.item())


def ties_merge_flat(
    base: torch.Tensor,
    task_vectors: Sequence[torch.Tensor],
    *,
    keep_fraction: float,
    alpha: float,
    disjoint: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Exact global TIES: trim, mass-sign elect, and disjoint merge."""
    if base.ndim != 1:
        raise ParameterFusionError("TIES base must be flat")
    vectors = _fp64_inputs(task_vectors)
    if any(value.ndim != 1 or value.shape != base.shape for value in vectors):
        raise ParameterFusionError("TIES task vectors differ from the flat base")
    if disjoint not in {"mean", "sum"}:
        raise ParameterFusionError("disjoint must be mean or sum")
    thresholds: list[float | None] = []
    trimmed: list[torch.Tensor] = []
    retained_counts: list[int] = []
    for vector in vectors:
        threshold = exact_global_keep_threshold(vector, keep_fraction)
        thresholds.append(threshold)
        selected = vector if threshold is None else vector * (vector.abs() >= threshold)
        trimmed.append(selected)
        retained_counts.append(int(torch.count_nonzero(selected).item()))
    elected = torch.sign(sum(trimmed))
    balance = int(elected.to(torch.int64).sum().item())
    zero_sign = 1 if balance > 0 else -1 if balance < 0 else 0
    if zero_sign:
        elected = torch.where(elected == 0, float(zero_sign), elected)
    selected_values = [
        vector * torch.where(elected > 0, vector > 0, vector < 0) for vector in trimmed
    ]
    merged = sum(selected_values)
    contributor_count = sum(value != 0 for value in selected_values)
    if disjoint == "mean":
        merged = merged / contributor_count.clamp_min(1)
    result = base.detach().cpu().double() + float(alpha) * merged
    metadata: dict[str, object] = {
        "thresholds": thresholds,
        "retained_counts": retained_counts,
        "merged_nonzero": int(torch.count_nonzero(merged).item()),
        "conflict_coordinates": int(
            torch.count_nonzero(
                (trimmed[0] != 0)
                & (trimmed[1] != 0)
                & (torch.sign(trimmed[0]) != torch.sign(trimmed[1]))
            ).item()
        )
        if len(trimmed) == 2
        else None,
        "zero_sign_fallback": zero_sign,
        "disjoint": disjoint,
        "keep_fraction": keep_fraction,
        "alpha": alpha,
    }
    return result, metadata


def dare_rescale_flat(
    task_vector: torch.Tensor,
    *,
    drop_probability: float,
    seed: int,
    chunk_size: int = 8 * 1024 * 1024,
) -> tuple[torch.Tensor, int]:
    """Deterministic DARE Bernoulli drop-and-rescale on one complete vector."""
    if task_vector.ndim != 1 or not task_vector.is_floating_point():
        raise ParameterFusionError("DARE task vector must be flat and floating")
    if not 0.0 <= drop_probability < 1.0:
        raise ParameterFusionError("drop_probability must lie in [0, 1)")
    keep_probability = 1.0 - drop_probability
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    source = task_vector.detach().cpu().double()
    output = torch.empty_like(source)
    retained = 0
    for start in range(0, source.numel(), chunk_size):
        end = min(start + chunk_size, source.numel())
        keep = torch.rand(end - start, generator=generator) < keep_probability
        retained += int(keep.sum().item())
        output[start:end] = source[start:end] * keep / keep_probability
    return output, retained


def dare_ties_merge_flat(
    base: torch.Tensor,
    task_vectors: Sequence[torch.Tensor],
    *,
    drop_probability: float,
    seed: int,
    alpha: float,
    disjoint: str = "mean",
) -> tuple[torch.Tensor, dict[str, object]]:
    """DARE sparsification followed by TIES sign election without a second trim."""
    sparse: list[torch.Tensor] = []
    counts: list[int] = []
    seeds: list[int] = []
    for index, vector in enumerate(task_vectors):
        task_seed = int(seed) + index
        transformed, retained = dare_rescale_flat(
            vector, drop_probability=drop_probability, seed=task_seed
        )
        sparse.append(transformed)
        counts.append(retained)
        seeds.append(task_seed)
    merged, metadata = ties_merge_flat(
        base,
        sparse,
        keep_fraction=1.0,
        alpha=alpha,
        disjoint=disjoint,
    )
    metadata.update(
        {
            "dare_drop_probability": drop_probability,
            "dare_keep_probability": 1.0 - drop_probability,
            "dare_seeds": seeds,
            "dare_retained_counts": counts,
            "second_ties_trim": False,
        }
    )
    return merged, metadata
