"""Validate execution provenance and the request/call structure before solving."""

import math
from collections import Counter
from pathlib import Path

METHOD = "pi05_full_vision_language_action_block_regmeanpp_replay_calibration"


def validate_trace(manifest, checkpoint, name, pass_index, cfg):
    if (
        manifest.get("task") != name
        or Path(manifest.get("calibration_policy", "")).resolve() != Path(checkpoint).resolve()
    ):
        raise ValueError(f"{name}: cache must identify the supplied frozen dense expert")
    samples = manifest.get("samples", [])
    if (
        manifest.get("method") != METHOD
        or not samples
        or len(samples) != manifest.get("sample_count")
    ):
        raise ValueError(f"{name}: incomplete full-prefix replay cache")
    source = manifest.get("source_kind", "expert_execution")
    is_demo = source.startswith("demonstration_observation")
    if is_demo != (cfg["variant"] == "demo"):
        raise ValueError("Demonstration inputs must be explicitly identified by variant=demo")
    if not is_demo and source not in ("expert_execution", "expert_execution_inputs"):
        raise ValueError("Only frozen-expert execution caches are accepted")
    groups = {}
    for i, sample in enumerate(samples):
        if sample.get("index", i) != i or sample.get("vision_count") != 3:
            raise ValueError("Expected ordered sample indices and all three camera inputs")
        if pass_index == 2 and "selected_request_slot" not in sample:
            raise ValueError("Cache B needs selected_request_slot metadata for row allocation")
        group = (
            sample["prompt_signature"],
            sample.get("episode_serial", 0),
            sample["request_index"],
        )
        groups.setdefault(group, []).append(sample["flow_index"])
    if any(sorted(flows) != [0, 5, 9] for flows in groups.values()):
        raise ValueError(
            "Keep full request/flow triples [0,5,9] in caches; final_call filtering happens in the solver"
        )
    if cfg["strict_paper_budget"]:
        counts = Counter(group[0] for group in groups)
        if len(samples) != 150 or len(counts) != 10 or set(counts.values()) != {5}:
            raise ValueError(
                "Paper recipe needs ten tasks, five requests/task and three calls/request"
            )


def validate_rows(metrics, names, pass_index, cfg, *, require_ridge=True):
    if len(metrics) != 418:
        raise ValueError(f"Expected 418 fitted modules, found {len(metrics)}")
    total = 0
    for name, entry in metrics.items():
        rows = entry["rows_by_expert"]
        if set(rows) != set(names) or any(n <= 0 for n in rows.values()):
            raise ValueError(f"Missing positive rows for {name}")
        if require_ridge and (not math.isfinite(entry["ridge"]) or entry["ridge"] <= 0):
            raise ValueError(f"Invalid ridge for {name}")
        if cfg["strict_paper_budget"]:
            time = name in ("model.time_mlp_in", "model.time_mlp_out")
            expected = (
                (150 if time else 4500 if ".vision_tower." in name else 1500)
                if pass_index == 1
                else (50 if time else 800)
            )
            if pass_index == 1 and cfg["variant"] == "final_call":
                expected //= 3
            if any(n != expected for n in rows.values()):
                raise ValueError(f"Row quota mismatch for {name}: {rows}, expected {expected}")
        total += sum(rows.values())
    return total
