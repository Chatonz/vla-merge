from types import SimpleNamespace

import pytest
from tcr_merging.calibration.collection import (
    BlockCalibrationCollector,
    CaptureConfig,
    ReplayCollector,
    environment,
)


def test_environment_restores_existing_and_unset_values(monkeypatch):
    monkeypatch.setenv("TCR_TEST_EXISTING", "before")
    monkeypatch.delenv("TCR_TEST_NEW", raising=False)
    import os

    with (
        pytest.raises(RuntimeError),
        environment({"TCR_TEST_EXISTING": "during", "TCR_TEST_NEW": 10}),
    ):
        assert os.environ["TCR_TEST_EXISTING"] == "during"
        assert os.environ["TCR_TEST_NEW"] == "10"
        raise RuntimeError("rollout failed")
    assert os.environ["TCR_TEST_EXISTING"] == "before"
    assert "TCR_TEST_NEW" not in os.environ


def test_collector_accepts_rollout_directory_but_refuses_cache_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(BlockCalibrationCollector, "__init__", lambda self, policy: None)
    policy = SimpleNamespace(
        config=SimpleNamespace(use_peft=False, num_inference_steps=10),
        model=SimpleNamespace(sample_noise=lambda: None),
    )
    output = tmp_path / "capture"
    (output / "rollout").mkdir(parents=True)
    config = CaptureConfig("test", tmp_path / "checkpoint", output)
    collector = ReplayCollector(policy, config)
    assert collector.capture_config.output == output
    (output / "replay.json").write_text("{}")
    with pytest.raises(FileExistsError):
        ReplayCollector(policy, config)


@pytest.mark.parametrize("fail", [False, True])
def test_generated_camera_validation_restores_configuration(monkeypatch, fail):
    factory = pytest.importorskip("lerobot.policies.factory")
    from tcr_merging.pi05.runtime import visual_feature_compatibility

    features = {"observation.images.image": 1, "observation.images.empty_camera_0": 2}
    cfg = SimpleNamespace(input_features=features)

    def original(config, available):
        assert config.input_features == {"observation.images.image": 1}
        if fail:
            raise ValueError("incompatible real camera")

    monkeypatch.setattr(factory, "validate_visual_features_consistency", original)
    with visual_feature_compatibility():
        if fail:
            with pytest.raises(ValueError):
                factory.validate_visual_features_consistency(cfg, {})
        else:
            factory.validate_visual_features_consistency(cfg, {})
        assert cfg.input_features is features
    assert factory.validate_visual_features_consistency is original
