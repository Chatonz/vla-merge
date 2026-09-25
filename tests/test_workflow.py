import json

import pytest
from tcr_merging.calibration.traces import validate_trace
from tcr_merging.cli import main
from tcr_merging.config import load_config
from tcr_merging.experiments import generate


def config_file(tmp_path, **updates):
    cfg = {
        "base_model": "base",
        "output": "output",
        "strict_paper_budget": False,
        "experts": {
            n: {"checkpoint": n, "cache_a": f"a/{n}", "cache_b": f"b/{n}"} for n in ("x", "y")
        },
        **updates,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_paths_relative_to_config_not_cwd(tmp_path, monkeypatch):
    p = config_file(tmp_path)
    monkeypatch.chdir("/")
    cfg = load_config(p)
    assert cfg["experts"]["x"]["cache_a"] == str(tmp_path / "a/x")


@pytest.mark.parametrize(
    "updates",
    [
        {"unknown": 1},
        {"ridge_ratio": -1},
        {"rows_per_call": 0},
        {"variant": "final_call"},
        {"first_pass_weighting": "other"},
    ],
)
def test_invalid_settings_fail_early(tmp_path, updates):
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, **updates))


def test_dry_run_does_not_need_weights(tmp_path, capsys):
    main(["merge", "--config", str(config_file(tmp_path)), "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert len(result["commands"]) == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "study,count", [("ridge", 3), ("budget", 3), ("passes", 2), ("weighting", 2)]
)
def test_study_configs_are_valid_and_never_launch(study, count, tmp_path):
    report = generate(config_file(tmp_path), study, tmp_path / "study")
    assert len(report["configs"]) == count
    for name in report["configs"]:
        load_config(tmp_path / "study" / f"{name}.json")
    assert not (tmp_path / "study/runs").exists()
    with pytest.raises(FileExistsError):
        generate(tmp_path / "config.json", study, tmp_path / "study")


def test_ablations_reference_full_but_keep_own_output(tmp_path):
    p = config_file(tmp_path)
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    demo = json.loads(p.read_text())
    demo["experts"] = {
        n: {"checkpoint": str(tmp_path / n), "cache_a": f"demo_a/{n}", "cache_b": f"demo_b/{n}"}
        for n in ("x", "y")
    }
    demo_path = demo_dir / "config.json"
    demo_path.write_text(json.dumps(demo))
    report = generate(p, "ablations", tmp_path / "study", demo_path)
    assert report["configs"] == ["matched_full", "final_call", "expert_prefix", "demo"]
    for name in report["configs"][1:]:
        cfg = load_config(tmp_path / "study" / f"{name}.json")
        assert cfg["ridge_reference"].endswith("matched_full/pass1")
        assert cfg["output"].endswith(name)


def test_trace_rejects_wrong_teacher_and_broken_flow_groups(tmp_path):
    cfg = {"variant": "full", "strict_paper_budget": False}
    m = {
        "task": "x",
        "calibration_policy": str(tmp_path),
        "sample_count": 3,
        "method": "pi05_full_vision_language_action_block_regmeanpp_replay_calibration",
        "samples": [
            {
                "index": i,
                "vision_count": 3,
                "prompt_signature": 5,
                "request_index": 0,
                "flow_index": f,
                "selected_request_slot": 0,
            }
            for i, f in enumerate((0, 5, 9))
        ],
    }
    validate_trace(m, tmp_path, "x", 2, cfg)
    with pytest.raises(ValueError):
        validate_trace(m, tmp_path / "wrong", "x", 2, cfg)
    m["samples"][1]["flow_index"] = 9
    with pytest.raises(ValueError):
        validate_trace(m, tmp_path, "x", 2, cfg)
