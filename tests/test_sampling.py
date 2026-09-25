import pytest
from tcr_merging.calibration.collection import select_quantiles
from tcr_merging.merging.prefix import advance_by_prefix
from tcr_merging.merging.sampling import row_indices
from tcr_merging.pi05.scope import module_groups, target_keys


@pytest.mark.parametrize("rows,cameras,cap", [(1, 1, 16), (50, 1, 16), (256, 3, 16), (2, 1, 24)])
def test_quota_is_shared_across_calls_and_views_without_duplicates(rows, cameras, cap):
    for slot in range(5):
        selected = [
            (camera, token)
            for flow in (0, 5, 9)
            for camera in range(cameras)
            for token in row_indices(rows, cap, flow, slot, cameras=cameras, camera=camera)
        ]
        assert len(selected) == min(cap, rows * cameras)
        assert len(selected) == len(set(selected))


def test_final_call_retains_request_quota():
    assert not row_indices(50, 16, 0, 0, last_only=True)
    assert len(row_indices(50, 16, 9, 0, last_only=True)) == 16


def test_prefix_modes_and_exception_restore():
    state = {"weight": 3}

    def expert(name):
        state["weight"] = {"a": 2, "b": 5}[name]

    def restore():
        state["weight"] = 3

    groups = {"a": [[]], "b": [[]]}
    advance_by_prefix(groups, "expert", expert, restore, lambda s: s.append(state["weight"]))
    assert groups == {"a": [[2]], "b": [[5]]}
    assert state["weight"] == 3
    advance_by_prefix(groups, "merged", expert, restore, lambda s: s.append(state["weight"]))
    assert groups == {"a": [[2, 3]], "b": [[5, 3]]}
    with pytest.raises(RuntimeError):
        advance_by_prefix(
            groups, "expert", expert, restore, lambda s: (_ for _ in ()).throw(RuntimeError())
        )
    assert state["weight"] == 3


def test_quantile_selection_keeps_complete_paths():
    metadata = [{"request_index": r, "flow_index": f} for r in range(13) for f in (0, 5, 9)]
    chosen = [metadata[i] for i in select_quantiles(metadata)]
    assert sorted({s["request_index"] for s in chosen}) == [0, 3, 6, 9, 12]
    assert len(chosen) == 15
    with pytest.raises(ValueError):
        select_quantiles(metadata[:6])


def test_paper_scope_is_explicit():
    assert len(target_keys()) == 422
    assert (
        sum(len(block) for family in module_groups().values() for block in family.values()) + 4
        == 418
    )
