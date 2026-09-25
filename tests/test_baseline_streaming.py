from __future__ import annotations

import itertools

import pytest
import torch
from tcr_merging.baselines.parameter_fusion import (
    dare_ties_merge_flat,
    exact_global_keep_threshold,
    ties_merge_flat,
)
from tcr_merging.baselines.streaming import (
    dare_ties_merge_flat_chunked,
    exact_global_keep_threshold_chunked,
    ties_merge_flat_chunked,
)


def _official_ties_direct_transcription(
    task_vectors: list[torch.Tensor],
    *,
    keep_fraction: float,
    alpha: float,
    disjoint: str,
) -> torch.Tensor:
    """Small-tensor transcription of the authors' merge_utils.py kernel."""
    matrix = torch.stack(task_vectors).double()
    retained = int(matrix.shape[1] * keep_fraction)
    if retained == matrix.shape[1]:
        trimmed = matrix
    else:
        kth = matrix.shape[1] - retained
        thresholds = matrix.abs().kthvalue(kth, dim=1, keepdim=True).values
        trimmed = matrix * (matrix.abs() >= thresholds)
    elected = torch.sign(trimmed.sum(dim=0))
    majority_sign = torch.sign(elected.sum())
    elected = torch.where(elected == 0, majority_sign, elected)
    selected = trimmed * torch.where(elected.unsqueeze(0) > 0, trimmed > 0, trimmed < 0)
    merged = selected.sum(dim=0)
    if disjoint == "mean":
        merged = merged / (selected != 0).sum(dim=0).clamp_min(1)
    elif disjoint != "sum":
        raise ValueError(disjoint)
    return merged * alpha


@pytest.mark.parametrize("length,keep", [(6, 0.5), (257, 0.2), (1025, 0.73)])
def test_radix_threshold_matches_reference(length: int, keep: float) -> None:
    generator = torch.Generator().manual_seed(20260906 + length)
    value = torch.randn(length, generator=generator)
    expected = exact_global_keep_threshold(value, keep)
    actual = exact_global_keep_threshold_chunked(value, keep, chunk_size=37)
    assert actual == expected


def test_radix_threshold_matches_reference_with_ties_and_signed_zero() -> None:
    value = torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 2.0, 3.0])
    assert exact_global_keep_threshold_chunked(
        value, 0.5, chunk_size=3
    ) == exact_global_keep_threshold(value, 0.5)


@pytest.mark.parametrize("disjoint", ["mean", "sum"])
def test_chunked_ties_matches_reference(disjoint: str) -> None:
    generator = torch.Generator().manual_seed(440)
    tasks = [torch.randn(503, generator=generator) for _ in range(4)]
    reference, reference_meta = ties_merge_flat(
        torch.zeros(503), tasks, keep_fraction=0.2, alpha=0.75, disjoint=disjoint
    )
    actual, actual_meta = ties_merge_flat_chunked(
        tasks, keep_fraction=0.2, alpha=0.75, disjoint=disjoint, chunk_size=71
    )
    assert torch.equal(actual.double(), reference)
    for key in ("thresholds", "retained_counts", "merged_nonzero", "zero_sign_fallback"):
        assert actual_meta[key] == reference_meta[key]


@pytest.mark.parametrize("disjoint", ["mean", "sum"])
def test_chunked_ties_matches_official_direct_transcription(disjoint: str) -> None:
    tasks = [
        torch.tensor([8.0, -7.0, 6.0, 0.1, 0.2, 0.3]),
        torch.tensor([-9.0, -5.0, 4.0, 0.3, 0.2, 0.1]),
        torch.tensor([1.0, 2.0, -3.0, 4.0, -5.0, 6.0]),
    ]
    expected = _official_ties_direct_transcription(
        tasks, keep_fraction=0.5, alpha=0.8, disjoint=disjoint
    )
    actual, _ = ties_merge_flat_chunked(
        tasks,
        keep_fraction=0.5,
        alpha=0.8,
        disjoint=disjoint,
        chunk_size=2,
    )
    assert torch.equal(actual, expected)


def test_chunked_ties_is_expert_order_invariant() -> None:
    tasks = [
        torch.tensor([8.0, -7.0, 6.0, 1.0, 2.0, 3.0]),
        torch.tensor([-9.0, -5.0, 4.0, 3.0, 2.0, 1.0]),
        torch.tensor([1.0, 2.0, -3.0, 4.0, -5.0, 6.0]),
    ]
    expected, _ = ties_merge_flat_chunked(
        tasks, keep_fraction=0.5, alpha=1.0, disjoint="mean", chunk_size=2
    )
    for permutation in itertools.permutations(tasks):
        actual, _ = ties_merge_flat_chunked(
            permutation,
            keep_fraction=0.5,
            alpha=1.0,
            disjoint="mean",
            chunk_size=2,
        )
        assert torch.equal(actual, expected)


def test_chunked_ties_is_chunk_invariant() -> None:
    tasks = [
        torch.tensor([8.0, -7.0, 6.0, 1.0, 2.0, 3.0, -4.0]),
        torch.tensor([-9.0, -5.0, 4.0, 3.0, 2.0, 1.0, 7.0]),
    ]
    expected, expected_meta = ties_merge_flat_chunked(
        tasks, keep_fraction=0.5, alpha=0.75, disjoint="mean", chunk_size=1
    )
    for chunk_size in (2, 3, 7, 99):
        actual, metadata = ties_merge_flat_chunked(
            tasks,
            keep_fraction=0.5,
            alpha=0.75,
            disjoint="mean",
            chunk_size=chunk_size,
        )
        assert torch.equal(actual, expected)
        for key in (
            "thresholds",
            "retained_counts",
            "merged_nonzero",
            "zero_sign_fallback",
        ):
            assert metadata[key] == expected_meta[key]


def test_chunked_ties_single_and_identical_experts() -> None:
    task = torch.tensor([1.0, -2.0, 0.25, 3.0])
    single, _ = ties_merge_flat_chunked(
        [task], keep_fraction=1.0, alpha=1.0, disjoint="mean", chunk_size=2
    )
    identical_mean, _ = ties_merge_flat_chunked(
        [task, task], keep_fraction=1.0, alpha=1.0, disjoint="mean", chunk_size=3
    )
    identical_sum, _ = ties_merge_flat_chunked(
        [task, task], keep_fraction=1.0, alpha=1.0, disjoint="sum", chunk_size=3
    )
    assert torch.equal(single, task.double())
    assert torch.equal(identical_mean, task.double())
    assert torch.equal(identical_sum, 2 * task.double())


@pytest.mark.parametrize("disjoint", ["mean", "sum"])
def test_chunked_dare_ties_matches_reference_for_one_rng_chunk(disjoint: str) -> None:
    generator = torch.Generator().manual_seed(900)
    tasks = [torch.randn(10_003, generator=generator) for _ in range(3)]
    reference, reference_meta = dare_ties_merge_flat(
        torch.zeros(10_003),
        tasks,
        drop_probability=0.9,
        seed=78,
        alpha=1.0,
        disjoint=disjoint,
    )
    actual, actual_meta = dare_ties_merge_flat_chunked(
        tasks,
        drop_probability=0.9,
        seed=78,
        alpha=1.0,
        disjoint=disjoint,
        chunk_size=20_000,
    )
    assert torch.equal(actual.double(), reference)
    assert actual_meta["dare_retained_counts"] == reference_meta["dare_retained_counts"]
    assert actual_meta["dare_seeds"] == reference_meta["dare_seeds"]
