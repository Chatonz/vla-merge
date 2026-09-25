"""Check imports/API compatibility without loading model weights or creating a simulator."""

import argparse
import importlib
import inspect
import json
import os
import sys
from importlib import metadata
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Require a visible CUDA device")
    parser.add_argument("--libero", action="store_true", help="Also check LIBERO/robosuite imports")
    args = parser.parse_args()
    expected = {
        "torch": "2.7.1",
        "torchvision": "0.22.1",
        "transformers": "5.5.4",
        "lerobot": "0.6.2",
        "peft": "0.20.0",
        "safetensors": "0.5.3",
    }
    report = {"python": sys.version.split()[0], "versions": {}, "warnings": [], "errors": []}
    for name, version in expected.items():
        try:
            actual = metadata.version(name)
            report["versions"][name] = actual
            if actual.split("+")[0] != version:
                report["warnings"].append(f"{name}: expected {version}, found {actual}")
        except metadata.PackageNotFoundError:
            report["errors"].append(f"Missing package: {name}")
    try:
        import torch
        from lerobot.policies import pi_gemma
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from tcr_merging.calibration.collection import ReplayCollector
        from tcr_merging.merging.engine import merge_pass
        from transformers.cache_utils import DynamicCache
        from transformers.models.gemma import modeling_gemma

        del PI05Policy, merge_pass, ReplayCollector
        for module, names in (
            (pi_gemma, ("layernorm_forward", "_gated_residual")),
            (modeling_gemma, ("apply_rotary_pos_emb", "eager_attention_forward")),
        ):
            for name in names:
                if not callable(getattr(module, name, None)):
                    raise RuntimeError(f"Missing replay interface: {module.__name__}.{name}")
        report["dynamic_cache_signature"] = str(inspect.signature(DynamicCache))
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if args.cuda and not report["cuda_available"]:
            report["errors"].append("CUDA requested but unavailable")
    except Exception as exc:
        report["errors"].append(f"Runtime import/API check: {type(exc).__name__}: {exc}")
    if args.libero:
        config = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")) / "config.yaml"
        names = ("libero", "robosuite", "mujoco", "lerobot.envs.libero")
        if not config.is_file():
            report["errors"].append(
                f"Missing LIBERO configuration: {config}. Prepare resources and config.yaml "
                "first; see docs/SETUP.md. Skipping imports that may prompt interactively."
            )
            names = ()
        for name in names:
            try:
                importlib.import_module(name)
            except Exception as exc:
                report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    report["status"] = (
        "failed" if report["errors"] else "passed_with_warnings" if report["warnings"] else "passed"
    )
    report["scope"] = "Imports and API presence only; not full replay or simulator validation."
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
