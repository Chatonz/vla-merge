"""Compatibility for pi0.5's generated (not dataset-provided) empty camera."""

from contextlib import contextmanager


@contextmanager
def visual_feature_compatibility():
    from lerobot.policies import factory

    original_validate = factory.validate_visual_features_consistency

    def validate(cfg, features):
        original = cfg.input_features
        cfg.input_features = {k: v for k, v in original.items() if ".empty_camera_" not in k}
        try:
            return original_validate(cfg, features)
        finally:
            cfg.input_features = original

    factory.validate_visual_features_consistency = validate
    try:
        yield
    finally:
        factory.validate_visual_features_consistency = original_validate
