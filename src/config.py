"""Portable paths and explicit experiment settings; no machine-specific defaults."""

import json
import math
from copy import deepcopy
from pathlib import Path

DEFAULTS = dict(
    device="cuda",
    passes=2,
    variant="full",
    ridge_ratio=0.05,
    max_correction_ratio=3.0,
    rows_per_call=10,
    rows_per_request=16,
    first_pass_weighting="prior",
    strict_paper_budget=True,
)
VARIANTS = ("full", "demo", "final_call", "expert_prefix")


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    allowed = {*DEFAULTS, "base_model", "experts", "output", "ridge_reference"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    cfg = {**DEFAULTS, **deepcopy(raw)}

    def resolve(value):
        p = Path(value).expanduser()
        return str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())

    for key in ("base_model", "output"):
        cfg[key] = resolve(cfg[key])
    if cfg.get("ridge_reference"):
        cfg["ridge_reference"] = resolve(cfg["ridge_reference"])
    experts = cfg.get("experts")
    if not isinstance(experts, dict) or len(experts) < 2:
        raise ValueError("At least two named experts are required")
    required = (
        {"checkpoint", "cache_a", "cache_b"} if cfg["passes"] == 2 else {"checkpoint", "cache_a"}
    )
    for name, entry in experts.items():
        if not name or name != name.strip().lower() or not isinstance(entry, dict):
            raise ValueError("Expert names must be nonempty lowercase strings")
        if not required <= set(entry) or set(entry) - {"checkpoint", "cache_a", "cache_b"}:
            raise ValueError(f"{name}: expected checkpoint, cache_a and (for two passes) cache_b")
        for key in entry:
            entry[key] = resolve(entry[key])
        if cfg["passes"] == 2 and entry["cache_a"] == entry["cache_b"]:
            raise ValueError(f"{name}: use separate A/B caches for this two-pass recipe")
    if cfg["variant"] not in VARIANTS or cfg["passes"] not in (1, 2):
        raise ValueError("Unknown variant or pass count")
    if cfg["variant"] != "full" and not cfg.get("ridge_reference"):
        raise ValueError(
            "Paired ablations require the matching Full pass1 directory as ridge_reference"
        )
    if cfg["first_pass_weighting"] not in ("prior", "none"):
        raise ValueError("first_pass_weighting must be prior or none")
    for key in ("ridge_ratio", "max_correction_ratio"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("rows_per_call", "rows_per_request"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["strict_paper_budget"] and (
        len(experts) != 4 or cfg["rows_per_call"] != 10 or cfg["rows_per_request"] != 16
    ):
        raise ValueError(
            "The paper budget requires four experts and row caps 10/16; disable strict_paper_budget for other budgets"
        )
    return cfg


def validate_inputs(cfg):
    from tcr_merging.calibration.traces import validate_trace
    from tcr_merging.pi05.checkpoints import validate_support

    roots = [Path(e["checkpoint"]) for e in cfg["experts"].values()]
    for root in [Path(cfg["base_model"]), *roots]:
        for name in ("model.safetensors", "config.json"):
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
        if json.loads((root / "config.json").read_text()).get("use_peft", False):
            raise ValueError(f"{root}: materialize the adapter as a dense checkpoint first")
    validate_support(roots)
    for name, e in cfg["experts"].items():
        for index, key in enumerate(("cache_a", "cache_b")[: cfg["passes"]], 1):
            cache = Path(e[key])
            if not (cache / "replay.safetensors").is_file():
                raise FileNotFoundError(cache / "replay.safetensors")
            manifest = json.loads((cache / "replay.json").read_text())
            validate_trace(manifest, e["checkpoint"], name, index, cfg)
    return cfg
