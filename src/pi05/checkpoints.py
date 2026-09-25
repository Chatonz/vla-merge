"""Checkpoint compatibility, support files and atomic output publication."""

import hashlib
import json
import shutil
from pathlib import Path

SUPPORT = (
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compatible_config(root):
    cfg = json.loads((root / "config.json").read_text())
    for key in ("pretrained_path", "device", "push_to_hub", "repo_id"):
        cfg.pop(key, None)
    return cfg


def validate_support(roots):
    first = roots[0]
    token_files = sorted(
        p.relative_to(first) for p in (first / "tokenizer").rglob("*") if p.is_file()
    )
    if not token_files:
        raise ValueError(f"Missing local tokenizer in {first}")
    reference = {str(p): sha256(first / p) for p in (*SUPPORT, *token_files)}
    for root in roots[1:]:
        if compatible_config(root) != compatible_config(first):
            raise ValueError(f"Incompatible policy configuration: {root}")
        other_tokens = sorted(
            p.relative_to(root) for p in (root / "tokenizer").rglob("*") if p.is_file()
        )
        if other_tokens != token_files or any(sha256(root / p) != h for p, h in reference.items()):
            raise ValueError(f"Normalization, processors or tokenizer differ: {root}")
    return reference


def copy_support(source, staging, destination):
    for name in SUPPORT:
        shutil.copy2(source / name, staging / name)
    shutil.copytree(source / "tokenizer", staging / "tokenizer")
    cfg = json.loads((source / "config.json").read_text())
    cfg.update(use_peft=False, pretrained_path=str(destination))
    (staging / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
