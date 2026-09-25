"""Ridge regression with the paper's relative-error or uniform expert masses."""

from __future__ import annotations

import math
from typing import Any

import torch


def task_loss(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor) -> float:
    residual = x @ (weight - target).T
    return float((residual * residual).mean())


def solve_weight_multi(
    inputs,
    weights,
    prior,
    ridge_ratio=0.05,
    ridge_scale="feature_energy",
    max_correction_ratio=3.0,
    expert_loss_normalization="none",
    fixed_ridge=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    names = list(inputs)
    if not names or prior.ndim != 2 or not torch.isfinite(prior).all():
        raise ValueError("Expected nonempty experts and a finite matrix prior")
    if ridge_scale not in ("feature_energy", "kernel_diagonal"):
        raise ValueError("Unknown ridge scale")
    if (
        not math.isfinite(ridge_ratio)
        or ridge_ratio <= 0
        or not math.isfinite(max_correction_ratio)
        or max_correction_ratio <= 0
    ):
        raise ValueError("Ridge ratio and correction cap must be positive and finite")
    if set(names) != set(weights):
        raise ValueError("Input/weight expert names differ")
    device = prior.device
    prior = prior.to(dtype=torch.float32)
    prepared_inputs: dict[str, torch.Tensor] = {}
    prepared_weights: dict[str, torch.Tensor] = {}
    row_counts: dict[str, int] = {}
    for name in names:
        x_i = inputs[name].to(device=device, dtype=torch.float32)
        w_i = weights[name].to(device=device, dtype=torch.float32)
        if (
            x_i.ndim != 2
            or x_i.shape[0] == 0
            or not torch.isfinite(x_i).all()
            or not torch.isfinite(w_i).all()
        ):
            raise ValueError(f"{name}: expected finite matrices and at least one feature row")
        if x_i.shape[1] != prior.shape[1] or w_i.shape != prior.shape:
            raise ValueError(
                f"{name}: activation/weight mismatch {tuple(x_i.shape)} {tuple(w_i.shape)}"
            )
        prepared_inputs[name] = x_i
        prepared_weights[name] = w_i
        row_counts[name] = int(x_i.shape[0])
    prior_loss_by_expert = {
        name: task_loss(prepared_inputs[name], prior, prepared_weights[name]) for name in names
    }
    if expert_loss_normalization == "none":
        base_objective_weights = {name: 1.0 / len(names) for name in names}
        normalization_floor = None
    elif expert_loss_normalization == "prior":
        mean_prior_loss = sum(prior_loss_by_expert.values()) / len(names)
        normalization_floor = max(mean_prior_loss * 1e-06, 1e-12)
        inverse_losses = {
            name: max(prior_loss_by_expert[name], normalization_floor) ** (-1.0) for name in names
        }
        inverse_sum = sum(inverse_losses.values())
        base_objective_weights = {name: inverse_losses[name] / inverse_sum for name in names}
    else:
        raise ValueError(f"Unsupported expert loss normalization: {expert_loss_normalization}")
    task_delta_reference = sum(
        (
            float(
                torch.linalg.vector_norm(
                    weights[name].to(device=device, dtype=torch.float32) - prior
                )
            )
            for name in names
        )
    ) / len(names)

    def solve_once(objective_weights: dict[str, float]) -> dict[str, Any]:
        scaled_x: list[torch.Tensor] = []
        scaled_targets: list[torch.Tensor] = []
        for name in names:
            x_i = prepared_inputs[name]
            w_i = prepared_weights[name]
            scale = math.sqrt(x_i.shape[0] / objective_weights[name])
            scaled_x.append(x_i / scale)
            scaled_targets.append(x_i @ (w_i - prior).T / scale)
        x = torch.cat(scaled_x, dim=0)
        residual_target = torch.cat(scaled_targets, dim=0)
        kernel = x @ x.T
        feature_energy = max(float((x * x).sum() / x.shape[1]), 1e-12)
        kernel_diagonal = max(float(torch.diagonal(kernel).mean()), 1e-12)
        ridge_reference = feature_energy if ridge_scale == "feature_energy" else kernel_diagonal
        ridge = ridge_ratio * ridge_reference if fixed_ridge is None else fixed_ridge
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("Fixed ridge must be positive and finite")
        kernel.diagonal().add_(ridge)
        cholesky, info = torch.linalg.cholesky_ex(kernel)
        if int(info.max()) != 0:
            raise RuntimeError(f"RegMean++ kernel is not positive definite; info={int(info.max())}")
        alpha = torch.cholesky_solve(residual_target, cholesky)
        correction = alpha.T @ x
        raw_correction_norm = float(torch.linalg.vector_norm(correction))
        trust_scale = 1.0
        limit = max_correction_ratio * max(task_delta_reference, 1e-12)
        if raw_correction_norm > limit:
            trust_scale = limit / raw_correction_norm
            correction = correction * trust_scale
        merged = prior + correction
        losses = {
            name: task_loss(prepared_inputs[name], merged, prepared_weights[name]) for name in names
        }
        relative_losses = {
            name: losses[name] / max(prior_loss_by_expert[name], 1e-12) for name in names
        }
        return {
            "merged": merged,
            "correction": correction,
            "losses": losses,
            "relative_losses": relative_losses,
            "robust_objective": max(relative_losses.values()),
            "objective_weights": dict(objective_weights),
            "feature_energy": feature_energy,
            "kernel_diagonal": kernel_diagonal,
            "ridge_reference": ridge_reference,
            "ridge": ridge,
            "kernel": kernel,
            "cholesky": cholesky,
            "trust_scale": trust_scale,
        }

    selected = solve_once(base_objective_weights)
    objective_weights = selected["objective_weights"]
    merged = selected["merged"]
    correction = selected["correction"]
    merged_loss_by_expert = selected["losses"]
    feature_energy = selected["feature_energy"]
    kernel_diagonal = selected["kernel_diagonal"]
    ridge_reference = selected["ridge_reference"]
    ridge = selected["ridge"]
    kernel = selected["kernel"]
    cholesky = selected["cholesky"]
    trust_scale = selected["trust_scale"]
    prior_loss = sum(prior_loss_by_expert.values()) / len(names)
    merged_loss = sum(merged_loss_by_expert.values()) / len(names)
    objective_prior_loss = sum(
        (objective_weights[name] * prior_loss_by_expert[name] for name in names)
    )
    objective_merged_loss = sum(
        (objective_weights[name] * merged_loss_by_expert[name] for name in names)
    )
    objective_tolerance = max(abs(objective_prior_loss) * 1e-06, 1e-12)
    rejected_nonimproving = False
    if kernel.shape[0] <= 8192:
        eigenvalues = torch.linalg.eigvalsh(kernel)
        kernel_condition = float(eigenvalues[-1] / eigenvalues[0])
        kernel_condition_method = "exact_eigvalsh"
    else:
        cholesky_diagonal = torch.diagonal(cholesky)
        kernel_condition = float((cholesky_diagonal.max() / cholesky_diagonal.min()).square())
        kernel_condition_method = "cholesky_diagonal_proxy"
    return (
        merged.detach().cpu(),
        {
            "expert_count": len(names),
            "rows_by_expert": row_counts,
            "expert_loss_normalization": expert_loss_normalization,
            "expert_loss_normalization_power": 1.0,
            "base_expert_objective_weights": base_objective_weights,
            "worst_relative_loss": max(
                (
                    merged_loss_by_expert[name] / max(prior_loss_by_expert[name], 1e-12)
                    for name in names
                )
            ),
            "rejected_nonimproving": rejected_nonimproving,
            "expert_objective_weights": objective_weights,
            "expert_normalization_floor": normalization_floor,
            "input_width": int(prior.shape[1]),
            "output_width": int(prior.shape[0]),
            "ridge": ridge,
            "ridge_scale": ridge_scale,
            "ridge_reference": ridge_reference,
            "feature_energy": feature_energy,
            "kernel_diagonal_mean": kernel_diagonal,
            "kernel_rows": int(kernel.shape[0]),
            "kernel_condition": kernel_condition,
            "kernel_condition_method": kernel_condition_method,
            "trust_scale": trust_scale,
            "correction_norm": float(torch.linalg.vector_norm(correction)),
            "prior_loss": prior_loss,
            "dense_regmean_loss": merged_loss,
            "objective_prior_loss": objective_prior_loss,
            "objective_dense_regmean_loss": objective_merged_loss,
            "objective_acceptance_tolerance": objective_tolerance,
            "objective_improvement_vs_prior": (objective_prior_loss - objective_merged_loss)
            / max(objective_prior_loss, 1e-12),
            "prior_loss_by_expert": prior_loss_by_expert,
            "dense_regmean_loss_by_expert": merged_loss_by_expert,
            "relative_improvement_by_expert": {
                name: (prior_loss_by_expert[name] - merged_loss_by_expert[name])
                / max(prior_loss_by_expert[name], 1e-12)
                for name in names
            },
            "improvement_vs_prior": (prior_loss - merged_loss) / max(prior_loss, 1e-12),
        },
    )
