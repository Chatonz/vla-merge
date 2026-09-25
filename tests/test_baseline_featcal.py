from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from tcr_merging.baselines.featcal import (
    FeatCalConfig,
    FeatCalError,
    featcal_layernorm_affine,
    featcal_linear_bias,
    featcal_linear_weight,
    interpolated_target_features,
    linear_feature_statistics,
    uniform_soup_tensor,
)

def _load_official_solver():
    reference = os.environ.get("FEATCAL_REFERENCE_SOLVER")
    if not reference:
        pytest.skip("Optional upstream comparison: set FEATCAL_REFERENCE_SOLVER to solver.py")
    path = Path(reference).expanduser()
    if not path.is_file():
        pytest.fail("FEATCAL_REFERENCE_SOLVER does not point to a readable solver.py")
    spec = importlib.util.spec_from_file_location("frozen_official_featcal_solver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture():
    dtype = torch.float64
    expert_weights = [
        torch.tensor([[1.5, -0.2], [0.3, 0.8]], dtype=dtype),
        torch.tensor([[0.4, 1.1], [-0.7, 1.6]], dtype=dtype),
    ]
    student = [
        torch.tensor([[1.0, 0.2], [0.4, 1.2], [-0.5, 0.7]], dtype=dtype),
        torch.tensor([[0.3, 1.1], [1.4, -0.2], [0.8, 0.6]], dtype=dtype),
    ]
    expert_inputs = [
        torch.tensor([[0.7, 0.4], [0.6, 1.0], [-0.1, 1.3]], dtype=dtype),
        torch.tensor([[0.5, 0.9], [1.0, 0.1], [1.1, 0.2]], dtype=dtype),
    ]
    config = FeatCalConfig(
        ridge_lambda=0.2,
        anchor_blend_rho=1.6,
        teacher_interp_alpha=0.3,
        covariance_eps=1e-10,
        solve_dtype=torch.float64,
    )
    stats = [
        linear_feature_statistics(s, e, alpha=config.teacher_interp_alpha)
        for s, e in zip(student, expert_inputs)
    ]
    soup = torch.stack(expert_weights).mean(dim=0)
    base = torch.tensor([[0.5, 0.1], [-0.2, 0.7]], dtype=dtype)
    return expert_weights, student, expert_inputs, stats, soup, base, config


def _weight_oracle(expert_weights, stats, soup, base, config):
    covariances = [row["cov_ss"] for row in stats]
    crosses = [row["cross_st"] for row in stats]
    norms = [torch.linalg.matrix_norm(cov, ord="fro") for cov in covariances]
    matrix = sum(cov / norm for cov, norm in zip(covariances, norms))
    matrix = matrix + (config.ridge_lambda + config.covariance_eps) * torch.eye(
        2, dtype=matrix.dtype
    )
    anchor = config.anchor_blend_rho * soup + (1 - config.anchor_blend_rho) * base
    rhs = (
        sum(weight @ cross.mT / norm for weight, cross, norm in zip(expert_weights, crosses, norms))
        + config.ridge_lambda * anchor
    )
    return torch.linalg.solve(matrix.mT, rhs.mT).mT


def test_feature_target_is_expert_student_interpolation() -> None:
    student = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expert = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    assert torch.equal(
        interpolated_target_features(student, expert, alpha=0.25),
        0.25 * expert + 0.75 * student,
    )


def test_linear_weight_matches_independent_paper_eq17_oracle() -> None:
    expert_weights, _, _, stats, soup, base, config = _fixture()
    observed, metadata = featcal_linear_weight(
        expert_weights,
        stats,
        soup_weight=soup,
        base_weight=base,
        config=config,
    )
    expected = _weight_oracle(expert_weights, stats, soup, base, config)
    assert torch.allclose(observed, expected, atol=1e-12, rtol=1e-12)
    assert metadata["task_normalization"] == "inverse_frobenius_covariance_norm"
    assert metadata["anchor_blend_rho"] == 1.6


def test_linear_weight_bias_and_layernorm_match_frozen_official_solver() -> None:
    official = _load_official_solver()
    expert_weights, _, _, stats, soup, base, _ = _fixture()
    expert_weights = [value.float() for value in expert_weights]
    stats = [
        {
            key: value.float() if isinstance(value, torch.Tensor) else value
            for key, value in row.items()
        }
        for row in stats
    ]
    soup = soup.float()
    base = base.float()
    config = FeatCalConfig(
        ridge_lambda=0.05,
        anchor_blend_rho=2.0,
        teacher_interp_alpha=0.3,
        covariance_eps=1e-8,
        solve_dtype=torch.float32,
    )
    observed_weight, _ = featcal_linear_weight(
        expert_weights, stats, soup_weight=soup, base_weight=base, config=config
    )
    expected_weight = official.featcal_merge_linear_weight(
        expert_weights,
        stats,
        ridge_lambda=config.ridge_lambda,
        merged_param=soup,
        base_param=base,
        anchor_blend_rho=config.anchor_blend_rho,
        covariance_eps=config.covariance_eps,
    )
    assert torch.equal(observed_weight, expected_weight)

    biases = [torch.tensor([0.2, -0.4]), torch.tensor([1.0, 0.3])]
    soup_bias = torch.stack(biases).mean(0)
    base_bias = torch.tensor([0.1, -0.2])
    observed_bias, _ = featcal_linear_bias(
        biases,
        expert_weights,
        stats,
        observed_weight,
        soup_bias=soup_bias,
        base_bias=base_bias,
        config=config,
    )
    expected_bias = official.featcal_merge_linear_bias(
        biases,
        expert_weights,
        stats,
        observed_weight,
        ridge_lambda=config.ridge_lambda,
        merged_param=soup_bias,
        base_param=base_bias,
        anchor_blend_rho=config.anchor_blend_rho,
        covariance_eps=config.covariance_eps,
    )
    assert torch.equal(observed_bias, expected_bias)

    gammas = [torch.tensor([1.2, 0.8]), torch.tensor([0.7, 1.4])]
    betas = [torch.tensor([0.1, -0.2]), torch.tensor([-0.3, 0.5])]
    ln_stats = [
        {"mean_student": torch.tensor([0.1, -0.1]), "energy_student": torch.tensor([1.1, 0.9])},
        {"mean_student": torch.tensor([-0.2, 0.2]), "energy_student": torch.tensor([0.8, 1.3])},
    ]
    soup_gamma = torch.stack(gammas).mean(0)
    soup_beta = torch.stack(betas).mean(0)
    base_gamma = torch.ones(2)
    base_beta = torch.zeros(2)
    observed_gamma, observed_beta, _ = featcal_layernorm_affine(
        gammas,
        betas,
        ln_stats,
        soup_weight=soup_gamma,
        base_weight=base_gamma,
        soup_bias=soup_beta,
        base_bias=base_beta,
        config=config,
    )
    expected_gamma, expected_beta = official.featcal_merge_layernorm_affine(
        gammas,
        betas,
        ln_stats,
        ridge_lambda=config.ridge_lambda,
        merged_weight=soup_gamma,
        base_weight=base_gamma,
        merged_bias=soup_beta,
        base_bias=base_beta,
        anchor_blend_rho=config.anchor_blend_rho,
        covariance_eps=config.covariance_eps,
    )
    assert torch.allclose(observed_gamma, expected_gamma, atol=1e-7, rtol=1e-7)
    assert torch.allclose(observed_beta, expected_beta, atol=1e-7, rtol=1e-7)


def test_task_covariance_normalization_removes_per_task_feature_scale() -> None:
    expert_weights, student, expert_inputs, stats, soup, base, config = _fixture()
    rescaled_stats = [
        linear_feature_statistics(
            student[0] * 7.0,
            expert_inputs[0] * 7.0,
            alpha=config.teacher_interp_alpha,
        ),
        stats[1],
    ]
    original, _ = featcal_linear_weight(
        expert_weights, stats, soup_weight=soup, base_weight=base, config=config
    )
    rescaled, _ = featcal_linear_weight(
        expert_weights,
        rescaled_stats,
        soup_weight=soup,
        base_weight=base,
        config=config,
    )
    assert torch.allclose(original, rescaled, atol=1e-12, rtol=1e-12)


def test_linear_bias_matches_independent_paper_eq55_oracle() -> None:
    expert_weights, _, _, stats, soup, base, config = _fixture()
    calibrated_weight, _ = featcal_linear_weight(
        expert_weights, stats, soup_weight=soup, base_weight=base, config=config
    )
    biases = [
        torch.tensor([0.2, -0.4], dtype=torch.float64),
        torch.tensor([1.0, 0.3], dtype=torch.float64),
    ]
    soup_bias = torch.stack(biases).mean(dim=0)
    base_bias = torch.tensor([0.1, -0.2], dtype=torch.float64)
    observed, metadata = featcal_linear_bias(
        biases,
        expert_weights,
        stats,
        calibrated_weight,
        soup_bias=soup_bias,
        base_bias=base_bias,
        config=config,
    )
    norms = torch.stack([row["cov_ss"].norm(p="fro") for row in stats])
    task_weights = norms.reciprocal()
    anchor = config.anchor_blend_rho * soup_bias + (1 - config.anchor_blend_rho) * base_bias
    numerator = config.ridge_lambda * anchor
    denominator = torch.tensor(config.ridge_lambda, dtype=torch.float64)
    for omega, bias, weight, row in zip(task_weights, biases, expert_weights, stats):
        numerator = numerator + omega * (
            bias + weight @ row["mean_teacher"] - calibrated_weight @ row["mean_student"]
        )
        denominator = denominator + omega
    expected = numerator / denominator
    assert torch.allclose(observed, expected, atol=1e-12, rtol=1e-12)
    assert metadata["bias_solve_after_weight"] is True


def test_single_identical_expert_is_fixed_point() -> None:
    weight = torch.tensor([[1.2, -0.3], [0.4, 0.8]], dtype=torch.float64)
    inputs = torch.tensor([[1.0, 0.2], [-0.4, 1.1], [0.6, 0.9]], dtype=torch.float64)
    stats = linear_feature_statistics(inputs, inputs, alpha=0.3)
    config = FeatCalConfig(
        ridge_lambda=0.4,
        anchor_blend_rho=2.0,
        covariance_eps=0.0,
        solve_dtype=torch.float64,
    )
    observed, _ = featcal_linear_weight(
        [weight], [stats], soup_weight=weight, base_weight=weight, config=config
    )
    assert torch.allclose(observed, weight, atol=1e-12, rtol=1e-12)


def test_task_order_is_invariant() -> None:
    expert_weights, _, _, stats, soup, base, config = _fixture()
    forward, _ = featcal_linear_weight(
        expert_weights, stats, soup_weight=soup, base_weight=base, config=config
    )
    reverse, _ = featcal_linear_weight(
        list(reversed(expert_weights)),
        list(reversed(stats)),
        soup_weight=soup,
        base_weight=base,
        config=config,
    )
    assert torch.allclose(forward, reverse, atol=1e-12, rtol=1e-12)


def test_layernorm_identity_solution() -> None:
    gamma = torch.tensor([1.5, 0.5], dtype=torch.float64)
    beta = torch.tensor([-0.25, 0.75], dtype=torch.float64)
    stats = {
        "mean_student": torch.zeros(2, dtype=torch.float64),
        "energy_student": torch.ones(2, dtype=torch.float64),
    }
    config = FeatCalConfig(ridge_lambda=0.2, covariance_eps=1e-12, solve_dtype=torch.float64)
    merged_gamma, merged_beta, metadata = featcal_layernorm_affine(
        [gamma],
        [beta],
        [stats],
        soup_weight=gamma,
        base_weight=gamma,
        soup_bias=beta,
        base_bias=beta,
        config=config,
    )
    assert torch.allclose(merged_gamma, gamma, atol=1e-12, rtol=1e-12)
    assert torch.allclose(merged_beta, beta, atol=1e-12, rtol=1e-12)
    assert metadata["strategy"] == "featcal_layernorm_eq59"


def test_uniform_soup_is_name_order_invariant() -> None:
    a = torch.tensor([1.0, 3.0])
    b = torch.tensor([5.0, -1.0])
    assert torch.equal(
        uniform_soup_tensor({"b": b, "a": a}),
        uniform_soup_tensor({"a": a, "b": b}),
    )
    assert torch.equal(uniform_soup_tensor({"a": a, "b": b}), (a + b) / 2)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("weight_shape", "shapes differ"),
        ("stats_shape", "shape differs"),
        ("nonfinite", "non-finite"),
    ],
)
def test_linear_solver_rejects_shape_and_finite_errors(mutation: str, message: str) -> None:
    expert_weights, _, _, stats, soup, base, config = _fixture()
    if mutation == "weight_shape":
        expert_weights[1] = torch.zeros(3, 2, dtype=torch.float64)
    elif mutation == "stats_shape":
        stats[1] = {**stats[1], "cov_ss": torch.zeros(3, 3, dtype=torch.float64)}
    else:
        stats[1] = {**stats[1], "cross_st": torch.full((2, 2), float("nan"), dtype=torch.float64)}
    with pytest.raises(FeatCalError, match=message):
        featcal_linear_weight(
            expert_weights, stats, soup_weight=soup, base_weight=base, config=config
        )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"ridge_lambda": -1.0}, "ridge_lambda"),
        ({"teacher_interp_alpha": 1.1}, "teacher_interp_alpha"),
        ({"anchor_blend_rho": float("nan")}, "anchor_blend_rho"),
        ({"covariance_eps": -1.0}, "covariance_eps"),
        ({"solve_dtype": torch.float16}, "solve_dtype"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(FeatCalError, match=message):
        FeatCalConfig(**kwargs)
