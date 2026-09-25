"""Pure CPU kernels for an audited FeatCal post-merge calibration port.

The functions implement FeatCal's published closed-form linear, bias, and
LayerNorm updates.  They consume already paired module-input statistics and do
not collect trajectories, choose teachers, or infer a calibration order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


class FeatCalError(ValueError):
    """Raised when a FeatCal equation or input contract is violated."""


@dataclass(frozen=True)
class FeatCalConfig:
    ridge_lambda: float = 0.05
    anchor_blend_rho: float = 2.0
    teacher_interp_alpha: float = 0.3
    covariance_eps: float = 1e-8
    solve_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if not math.isfinite(self.ridge_lambda) or self.ridge_lambda < 0:
            raise FeatCalError("ridge_lambda must be finite and nonnegative")
        if not math.isfinite(self.anchor_blend_rho):
            raise FeatCalError("anchor_blend_rho must be finite")
        if not math.isfinite(self.teacher_interp_alpha) or not (
            0 <= self.teacher_interp_alpha <= 1
        ):
            raise FeatCalError("teacher_interp_alpha must lie in [0, 1]")
        if not math.isfinite(self.covariance_eps) or self.covariance_eps < 0:
            raise FeatCalError("covariance_eps must be finite and nonnegative")
        if self.solve_dtype not in {torch.float32, torch.float64}:
            raise FeatCalError("solve_dtype must be float32 or float64")


def _check_tensor(value: torch.Tensor, label: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise FeatCalError(f"{label} must be a tensor")
    if not value.is_floating_point():
        raise FeatCalError(f"{label} must be floating point")
    if value.numel() and not bool(torch.isfinite(value).all()):
        raise FeatCalError(f"{label} contains non-finite values")


def _validate_same_shape(values: Sequence[torch.Tensor], label: str) -> None:
    if not values:
        raise FeatCalError(f"{label} must not be empty")
    shape = values[0].shape
    for index, value in enumerate(values):
        _check_tensor(value, f"{label}[{index}]")
        if value.shape != shape:
            raise FeatCalError(f"{label} shapes differ")


def interpolated_target_features(
    student_inputs: torch.Tensor,
    expert_inputs: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    _check_tensor(student_inputs, "student_inputs")
    _check_tensor(expert_inputs, "expert_inputs")
    if student_inputs.shape != expert_inputs.shape or student_inputs.ndim != 2:
        raise FeatCalError("paired student/expert inputs must be equal-shape matrices")
    if not math.isfinite(alpha) or not (0 <= alpha <= 1):
        raise FeatCalError("alpha must lie in [0, 1]")
    return alpha * expert_inputs + (1.0 - alpha) * student_inputs


def linear_feature_statistics(
    student_inputs: torch.Tensor,
    expert_inputs: torch.Tensor,
    *,
    alpha: float,
) -> dict[str, torch.Tensor | int]:
    """Build row-oriented statistics corresponding to paper Eqs. 10 and 13."""
    target_inputs = interpolated_target_features(student_inputs, expert_inputs, alpha=alpha)
    rows = int(student_inputs.shape[0])
    if rows < 1:
        raise FeatCalError("feature statistics require at least one paired row")
    return {
        "cov_ss": student_inputs.mT @ student_inputs / rows,
        "cross_st": student_inputs.mT @ target_inputs / rows,
        "mean_student": student_inputs.mean(dim=0),
        "mean_teacher": target_inputs.mean(dim=0),
        "rows": rows,
    }


def _anchor(
    soup: torch.Tensor,
    base: torch.Tensor,
    rho: float,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if soup.shape != base.shape:
        raise FeatCalError("soup/base anchor shapes differ")
    _check_tensor(soup, "soup anchor")
    _check_tensor(base, "base anchor")
    return rho * soup.to(dtype=dtype) + (1.0 - rho) * base.to(dtype=dtype)


def _linear_stats(
    stats: Mapping[str, torch.Tensor | int],
    input_dim: int,
    *,
    dtype: torch.dtype,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("cov_ss", "cross_st", "mean_student", "mean_teacher")
    missing = [key for key in required if key not in stats]
    if missing:
        raise FeatCalError(f"{label} is missing statistics: {missing}")
    cov = stats["cov_ss"]
    cross = stats["cross_st"]
    mean_student = stats["mean_student"]
    mean_teacher = stats["mean_teacher"]
    assert isinstance(cov, torch.Tensor)
    assert isinstance(cross, torch.Tensor)
    assert isinstance(mean_student, torch.Tensor)
    assert isinstance(mean_teacher, torch.Tensor)
    for name, value in (
        ("cov_ss", cov),
        ("cross_st", cross),
        ("mean_student", mean_student),
        ("mean_teacher", mean_teacher),
    ):
        _check_tensor(value, f"{label}/{name}")
    if cov.shape != (input_dim, input_dim) or cross.shape != (input_dim, input_dim):
        raise FeatCalError(f"{label} covariance/cross shape differs from input width")
    if mean_student.shape != (input_dim,) or mean_teacher.shape != (input_dim,):
        raise FeatCalError(f"{label} feature mean shape differs from input width")
    return (
        cov.to(dtype=dtype),
        cross.to(dtype=dtype),
        mean_student.to(dtype=dtype),
        mean_teacher.to(dtype=dtype),
    )


def featcal_linear_weight(
    expert_weights: Sequence[torch.Tensor],
    task_stats: Sequence[Mapping[str, torch.Tensor | int]],
    *,
    soup_weight: torch.Tensor,
    base_weight: torch.Tensor,
    config: FeatCalConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Solve paper Eq. 17 using the official row-oriented implementation."""
    _validate_same_shape(expert_weights, "expert_weights")
    if len(expert_weights) != len(task_stats):
        raise FeatCalError("expert_weights and task_stats counts differ")
    if expert_weights[0].ndim != 2:
        raise FeatCalError("linear expert weights must be 2-D")
    if soup_weight.shape != expert_weights[0].shape or base_weight.shape != soup_weight.shape:
        raise FeatCalError("linear soup/base/expert shapes differ")
    out_dtype = soup_weight.dtype
    dtype = config.solve_dtype
    device = expert_weights[0].device
    input_dim = int(expert_weights[0].shape[1])
    parsed = [
        _linear_stats(stats, input_dim, dtype=dtype, label=f"task_stats[{index}]")
        for index, stats in enumerate(task_stats)
    ]
    covariances = [row[0].to(device=device) for row in parsed]
    crosses = [row[1].to(device=device) for row in parsed]
    norms = torch.stack([cov.norm(p="fro").clamp_min(config.covariance_eps) for cov in covariances])
    if config.covariance_eps == 0 and bool(torch.any(norms == 0)):
        raise FeatCalError("zero covariance norm requires positive covariance_eps")
    normalized_covariances = [cov / norm for cov, norm in zip(covariances, norms)]
    normalized_crosses = [cross / norm for cross, norm in zip(crosses, norms)]
    identity = torch.eye(input_dim, dtype=dtype, device=device)
    solve_matrix = torch.stack(normalized_covariances).sum(dim=0)
    solve_matrix = solve_matrix + (config.ridge_lambda + config.covariance_eps) * identity
    processed_weights = [value.to(device=device, dtype=dtype).mT for value in expert_weights]
    rhs = torch.stack(
        [cross @ weight for cross, weight in zip(normalized_crosses, processed_weights)]
    ).sum(dim=0)
    anchor = _anchor(
        soup_weight,
        base_weight,
        config.anchor_blend_rho,
        dtype=dtype,
    ).to(device=device)
    if config.ridge_lambda > 0:
        rhs = rhs + config.ridge_lambda * anchor.mT
    solver = "solve"
    try:
        solved = torch.linalg.solve(solve_matrix, rhs)
    except RuntimeError:
        solved = torch.linalg.pinv(solve_matrix) @ rhs
        solver = "pinv"
    result = solved.mT.to(dtype=out_dtype)
    _check_tensor(result, "calibrated linear weight")
    return result, {
        "strategy": "featcal_linear_eq17",
        "solver": solver,
        "task_count": len(expert_weights),
        "task_covariance_norms": [float(value.item()) for value in norms],
        "ridge_lambda": config.ridge_lambda,
        "anchor_blend_rho": config.anchor_blend_rho,
        "task_normalization": "inverse_frobenius_covariance_norm",
    }


def featcal_linear_bias(
    expert_biases: Sequence[torch.Tensor],
    expert_weights: Sequence[torch.Tensor],
    task_stats: Sequence[Mapping[str, torch.Tensor | int]],
    calibrated_weight: torch.Tensor,
    *,
    soup_bias: torch.Tensor,
    base_bias: torch.Tensor,
    config: FeatCalConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Solve the second-stage bias update in paper Eq. 55."""
    _validate_same_shape(expert_biases, "expert_biases")
    _validate_same_shape(expert_weights, "expert_weights")
    if not (len(expert_biases) == len(expert_weights) == len(task_stats)):
        raise FeatCalError("bias, weight, and task_stats counts differ")
    if expert_biases[0].ndim != 1 or expert_weights[0].ndim != 2:
        raise FeatCalError("linear bias/weight ranks are invalid")
    if expert_weights[0].shape[0] != expert_biases[0].shape[0]:
        raise FeatCalError("linear bias width differs from weight output width")
    if calibrated_weight.shape != expert_weights[0].shape:
        raise FeatCalError("calibrated weight shape differs")
    if soup_bias.shape != expert_biases[0].shape or base_bias.shape != soup_bias.shape:
        raise FeatCalError("bias soup/base/expert shapes differ")

    dtype = config.solve_dtype
    device = expert_weights[0].device
    input_dim = int(expert_weights[0].shape[1])
    parsed = [
        _linear_stats(stats, input_dim, dtype=dtype, label=f"task_stats[{index}]")
        for index, stats in enumerate(task_stats)
    ]
    norms = torch.stack(
        [row[0].to(device=device).norm(p="fro").clamp_min(config.covariance_eps) for row in parsed]
    )
    if config.covariance_eps == 0 and bool(torch.any(norms == 0)):
        raise FeatCalError("zero covariance norm requires positive covariance_eps")
    weights = norms.reciprocal()
    calibrated = calibrated_weight.to(device=device, dtype=dtype)
    numerator = torch.zeros_like(expert_biases[0], device=device, dtype=dtype)
    denominator = torch.zeros((), device=device, dtype=dtype)
    for task_weight, bias, weight, row in zip(weights, expert_biases, expert_weights, parsed):
        mean_student = row[2].to(device=device)
        mean_target = row[3].to(device=device)
        numerator = numerator + task_weight * (
            bias.to(device=device, dtype=dtype)
            + weight.to(device=device, dtype=dtype) @ mean_target
            - calibrated @ mean_student
        )
        denominator = denominator + task_weight
    anchor = _anchor(
        soup_bias,
        base_bias,
        config.anchor_blend_rho,
        dtype=dtype,
    ).to(device=device)
    if config.ridge_lambda > 0:
        numerator = numerator + config.ridge_lambda * anchor
        denominator = denominator + config.ridge_lambda
    result = (numerator / denominator.clamp_min(torch.finfo(dtype).eps)).to(dtype=soup_bias.dtype)
    _check_tensor(result, "calibrated linear bias")
    return result, {
        "strategy": "featcal_linear_bias_eq55",
        "task_count": len(expert_biases),
        "bias_solve_after_weight": True,
        "task_weights": [float(value.item()) for value in weights],
    }


def featcal_layernorm_affine(
    expert_weights: Sequence[torch.Tensor],
    expert_biases: Sequence[torch.Tensor],
    task_stats: Sequence[Mapping[str, torch.Tensor | int]],
    *,
    soup_weight: torch.Tensor,
    base_weight: torch.Tensor,
    soup_bias: torch.Tensor,
    base_bias: torch.Tensor,
    config: FeatCalConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Solve the coordinate-wise LayerNorm affine update in paper Eq. 59."""
    _validate_same_shape(expert_weights, "layernorm expert weights")
    _validate_same_shape(expert_biases, "layernorm expert biases")
    if not (len(expert_weights) == len(expert_biases) == len(task_stats)):
        raise FeatCalError("LayerNorm weight, bias, and task_stats counts differ")
    shape = expert_weights[0].shape
    if len(shape) != 1 or expert_biases[0].shape != shape:
        raise FeatCalError("LayerNorm affine parameters must be same-shape vectors")
    for anchor_name, value in (
        ("soup_weight", soup_weight),
        ("base_weight", base_weight),
        ("soup_bias", soup_bias),
        ("base_bias", base_bias),
    ):
        _check_tensor(value, anchor_name)
        if value.shape != shape:
            raise FeatCalError("LayerNorm anchor shapes differ")

    dtype = config.solve_dtype
    device = expert_weights[0].device
    mean_sum = torch.zeros(shape, device=device, dtype=dtype)
    energy_sum = torch.zeros(shape, device=device, dtype=dtype)
    rhs_weight = torch.zeros(shape, device=device, dtype=dtype)
    rhs_bias = torch.zeros(shape, device=device, dtype=dtype)
    for index, (gamma, beta, stats) in enumerate(zip(expert_weights, expert_biases, task_stats)):
        mean = stats.get("mean_student")
        energy = stats.get("energy_student")
        if not isinstance(mean, torch.Tensor) or not isinstance(energy, torch.Tensor):
            raise FeatCalError(f"LayerNorm stats[{index}] are incomplete")
        _check_tensor(mean, f"LayerNorm stats[{index}]/mean_student")
        _check_tensor(energy, f"LayerNorm stats[{index}]/energy_student")
        if mean.shape != shape or energy.shape != shape:
            raise FeatCalError("LayerNorm statistic shapes differ")
        mean = mean.to(device=device, dtype=dtype)
        energy = energy.to(device=device, dtype=dtype)
        gamma = gamma.to(device=device, dtype=dtype)
        beta = beta.to(device=device, dtype=dtype)
        mean_sum += mean
        energy_sum += energy
        rhs_weight += energy * gamma + mean * beta
        rhs_bias += mean * gamma + beta

    weight_anchor = _anchor(soup_weight, base_weight, config.anchor_blend_rho, dtype=dtype).to(
        device=device
    )
    bias_anchor = _anchor(soup_bias, base_bias, config.anchor_blend_rho, dtype=dtype).to(
        device=device
    )
    task_count = float(len(expert_weights))
    a11 = energy_sum + config.ridge_lambda
    a12 = mean_sum
    a22 = task_count + config.ridge_lambda
    rhs_weight += config.ridge_lambda * weight_anchor
    rhs_bias += config.ridge_lambda * bias_anchor
    determinant = (a11 * a22 - a12.square()).clamp_min(config.covariance_eps)
    weight = ((a22 * rhs_weight - a12 * rhs_bias) / determinant).to(dtype=soup_weight.dtype)
    bias = ((a11 * rhs_bias - a12 * rhs_weight) / determinant).to(dtype=soup_bias.dtype)
    _check_tensor(weight, "calibrated LayerNorm weight")
    _check_tensor(bias, "calibrated LayerNorm bias")
    return (
        weight,
        bias,
        {
            "strategy": "featcal_layernorm_eq59",
            "task_count": len(expert_weights),
            "equal_task_weights": True,
        },
    )


def uniform_soup_tensor(
    expert_values: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return a deterministic task-name-sorted uniform Soup tensor."""
    if not expert_values:
        raise FeatCalError("uniform Soup requires at least one expert")
    names = sorted(expert_values)
    values = [expert_values[name] for name in names]
    _validate_same_shape(values, "uniform Soup expert values")
    dtype = values[0].dtype
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    result = torch.stack([value.to(dtype=work_dtype) for value in values]).mean(dim=0)
    return result.to(dtype=dtype)


__all__ = [
    "FeatCalConfig",
    "FeatCalError",
    "featcal_layernorm_affine",
    "featcal_linear_bias",
    "featcal_linear_weight",
    "interpolated_target_features",
    "linear_feature_statistics",
    "uniform_soup_tensor",
]
