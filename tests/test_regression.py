import pytest
import torch
from tcr_merging.merging.regression import solve_weight_multi


@pytest.mark.parametrize("mode", ["none", "prior"])
@pytest.mark.parametrize("fixed_ridge", [None, 0.17])
def test_dual_solve_matches_independent_normal_equations(mode, fixed_ridge):
    generator = torch.Generator().manual_seed(17)
    prior = torch.randn(3, 5, generator=generator)
    weights = {n: torch.randn(3, 5, generator=generator) for n in ("a", "b")}
    inputs = {
        "a": torch.randn(7, 5, generator=generator),
        "b": torch.randn(4, 5, generator=generator),
    }
    before = prior.clone()
    actual, metrics = solve_weight_multi(
        inputs,
        weights,
        prior,
        max_correction_ratio=100,
        expert_loss_normalization=mode,
        fixed_ridge=fixed_ridge,
    )
    gram = torch.zeros(5, 5, dtype=torch.float64)
    rhs = torch.zeros(5, 3, dtype=torch.float64)
    for name, x in inputs.items():
        x = x.double()
        mass = metrics["expert_objective_weights"][name]
        g = x.T @ x * (mass / len(x))
        gram += g
        rhs += g @ (weights[name].double() - prior.double()).T
    expected = prior.double() + torch.linalg.solve(gram + metrics["ridge"] * torch.eye(5), rhs).T
    torch.testing.assert_close(actual.double(), expected, rtol=3e-5, atol=3e-5)
    assert torch.equal(prior, before)
    if fixed_ridge is not None:
        assert metrics["ridge"] == fixed_ridge


def test_unobserved_direction_retains_prior():
    x = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    prior = torch.tensor([[0.5, 4.0, -2.0]])
    result, _ = solve_weight_multi(
        {"a": x, "b": x}, {"a": torch.ones(1, 3), "b": torch.zeros(1, 3)}, prior
    )
    assert torch.equal(result[:, 1:], prior[:, 1:])


def test_relative_masses_favor_lower_prior_error():
    x = torch.eye(2)
    p = torch.zeros(1, 2)
    _, metrics = solve_weight_multi(
        {"a": x, "b": x},
        {"a": torch.ones(1, 2), "b": torch.ones(1, 2) * 2},
        p,
        expert_loss_normalization="prior",
    )
    assert metrics["expert_objective_weights"] == pytest.approx({"a": 0.8, "b": 0.2})


def test_identical_experts_and_zero_features_are_finite():
    p = torch.ones(2, 3)
    result, _ = solve_weight_multi(
        {"a": torch.zeros(4, 3), "b": torch.zeros(2, 3)},
        {"a": p, "b": p},
        p,
        expert_loss_normalization="prior",
    )
    assert torch.equal(result, p)


def test_correction_safeguard():
    p = torch.zeros(1, 2)
    weights = {"a": torch.ones(1, 2), "b": torch.ones(1, 2) * 2}
    x = {"a": torch.eye(2), "b": torch.eye(2)}
    result, info = solve_weight_multi(x, weights, p, max_correction_ratio=0.01)
    limit = 0.01 * sum(torch.linalg.vector_norm(w - p) for w in weights.values()) / 2
    assert torch.linalg.vector_norm(result - p) <= limit + 1e-7
    assert info["trust_scale"] < 1


@pytest.mark.parametrize("bad", [torch.empty(0, 2), torch.tensor([[float("nan"), 1.0]])])
def test_invalid_features_rejected(bad):
    with pytest.raises(ValueError):
        solve_weight_multi({"a": bad}, {"a": torch.ones(1, 2)}, torch.zeros(1, 2))
