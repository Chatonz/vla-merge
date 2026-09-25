"""Generate a small, explicit experiment matrix; no cluster scheduler."""

import json
import sys
from copy import deepcopy
from pathlib import Path

from tcr_merging.config import load_config


def generate(config_path, study, output, demo_config=None):
    base = load_config(config_path)
    if base["variant"] != "full" or base["passes"] != 2 or base.get("ridge_reference"):
        raise ValueError(
            "Start a study from a two-pass Full configuration with no external ridge reference"
        )
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    configs = {}

    def add(name, **updates):
        cfg = deepcopy(base)
        cfg.update(updates, output=str(output / "runs" / name))
        configs[name] = cfg

    if study == "ablations":
        add("matched_full")
        reference = str(output / "runs/matched_full/pass1")
        add("final_call", variant="final_call", ridge_reference=reference)
        add("expert_prefix", variant="expert_prefix", ridge_reference=reference)
        if demo_config is None:
            raise ValueError(
                "The ablation matrix requires --demo-config with separate demonstration caches"
            )
        demo = load_config(demo_config)
        if {n: e["checkpoint"] for n, e in demo["experts"].items()} != {
            n: e["checkpoint"] for n, e in base["experts"].items()
        }:
            raise ValueError("Demo and execution experts must match")
        add("demo", variant="demo", ridge_reference=reference, experts=demo["experts"])
    elif study == "ridge":
        for value in (0.01, 0.05, 0.1):
            add(f"ridge_{value:g}", ridge_ratio=value)
    elif study == "budget":
        add("reference")
        for value in (8, 24):
            add(
                f"rows_{value}",
                rows_per_request=value,
                strict_paper_budget=False,
                ridge_reference=str(output / "runs/reference/pass1"),
            )
    elif study == "passes":
        add("one_pass", passes=1)
        add("two_pass", passes=2)
    elif study == "weighting":
        add("relative")
        add(
            "uniform",
            first_pass_weighting="none",
            ridge_reference=str(output / "runs/relative/pass1"),
        )
    else:
        raise ValueError(f"Unknown study: {study}")
    output.mkdir(parents=True)
    commands = []
    for name, cfg in configs.items():
        path = output / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        commands.append([sys.executable, "-m", "tcr_merging.cli", "merge", "--config", str(path)])
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    return {
        "study": study,
        "configs": list(configs),
        "commands": commands,
        "note": "Exploratory sweeps are not new paper results. Run commands in order, then evaluate every arm on shared held-out resets.",
    }
