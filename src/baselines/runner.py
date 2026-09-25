"""Portable dense pi0.5 baseline entry points. Never connects to a robot."""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tcr_merging.config import load_config
from tcr_merging.pi05.checkpoints import copy_support, sha256, validate_support
from tcr_merging.pi05.scope import target_keys


def parameter_merge(cfg, method, output, *, keep_fraction=0.2, alpha=1.0):
    """Dense equal mean or global-domain TIES; calibration is not used."""
    from tcr_merging.baselines.streaming import ties_merge_flat_chunked

    if method not in ("soup", "ties"):
        raise ValueError("Unknown parameter baseline")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    experts = {n: Path(e["checkpoint"]) for n, e in cfg["experts"].items()}
    validate_support(list(experts.values()))
    base_root = Path(cfg["base_model"])
    adapted = sorted(target_keys())
    with ExitStack() as stack:
        base = stack.enter_context(
            safe_open(base_root / "model.safetensors", framework="pt", device="cpu")
        )
        sources = {
            n: stack.enter_context(safe_open(p / "model.safetensors", framework="pt", device="cpu"))
            for n, p in experts.items()
        }
        keys = set(base.keys())
        if not set(adapted) <= keys or any(set(s.keys()) != keys for s in sources.values()):
            raise ValueError("Base/expert schema differs from the supported pi0.5 scope")
        for root in [base_root, *experts.values()]:
            if json.loads((root / "config.json").read_text()).get("use_peft", False):
                raise ValueError("Materialize LoRA as a dense checkpoint first")
        tensors, vectors, shapes = {}, {n: [] for n in experts}, {}
        for key in sorted(keys):
            b = base.get_tensor(key)
            values = [source.get_tensor(key) for source in sources.values()]
            if not torch.isfinite(b).all() or any(
                v.shape != b.shape or v.dtype != b.dtype or not torch.isfinite(v).all()
                for v in values
            ):
                raise ValueError(f"Incompatible or nonfinite tensor: {key}")
            if key not in adapted:
                if any(not torch.equal(v, b) for v in values):
                    raise ValueError(f"Adaptation outside supported scope: {key}")
                tensors[key] = b
            elif method == "soup":
                tensors[key] = (sum(v.float() for v in values) / len(values)).to(b.dtype)
            else:
                tensors[key] = b
                shapes[key] = b.shape
                for name, v in zip(experts, values, strict=True):
                    vectors[name].append((v.float() - b.float()).reshape(-1))
        info = {"weighting": "uniform", "calibration_used": False}
        if method == "ties":
            # Sorted full scope, not independent per-layer trimming.
            flat = [torch.cat(vectors.pop(n)) for n in experts]
            delta, info = ties_merge_flat_chunked(flat, keep_fraction=keep_fraction, alpha=alpha)
            del flat
            offset = 0
            for key in adapted:
                count = tensors[key].numel()
                tensors[key] = (
                    tensors[key].double() + delta[offset : offset + count].reshape(shapes[key])
                ).to(tensors[key].dtype)
                offset += count
        if any(not torch.isfinite(v).all() for v in tensors.values()):
            raise ValueError("Merged tensor overflow; no output published")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".baseline-staging-", dir=output.parent))
        copy_support(next(iter(experts.values())), staging, output)
        save_file({k: v.contiguous() for k, v in tensors.items()}, staging / "model.safetensors")
    report = dict(
        method=method,
        created_at=datetime.now(timezone.utc).isoformat(),
        base_model_sha256=sha256(base_root / "model.safetensors"),
        expert_sha256={n: sha256(p / "model.safetensors") for n, p in experts.items()},
        model_sha256=sha256(staging / "model.safetensors"),
        modified_tensor_count=len(adapted),
        config=cfg,
        settings=info,
    )
    (staging / "baseline_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    staging.rename(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("soup", "ties", "regmeanpp", "featcal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ties-keep-fraction", type=float, default=0.2)
    parser.add_argument("--ties-alpha", type=float, default=1.0)
    parser.add_argument("--regmeanpp-solver", choices=("adapted", "original"), default="adapted")
    parser.add_argument("--offdiag-scale", type=float, default=0.95)
    parser.add_argument("--featcal-lambda", type=float, default=0.05)
    parser.add_argument("--featcal-rho", type=float, default=2.0)
    parser.add_argument("--featcal-alpha", type=float, default=0.3)
    # Internal subprocess phases separate teacher and student memory lifetimes.
    parser.add_argument("--phase", choices=("teacher", "student"), help=argparse.SUPPRESS)
    parser.add_argument("--features", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.phase is not None or args.features is not None) and args.method != "featcal":
        raise ValueError("Internal teacher/student phases are FeatCal-only")
    cfg = load_config(args.config)
    if cfg["variant"] != "full" or cfg.get("ridge_reference"):
        raise ValueError("Baseline entry expects a Full execution-cache config, not a TCR ablation")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    for name in (
        "ties_alpha",
        "featcal_lambda",
        "featcal_rho",
        "featcal_alpha",
        "offdiag_scale",
        "ties_keep_fraction",
    ):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"Nonfinite setting: {name}")
    if not 0 < args.ties_keep_fraction <= 1 or not 0 < args.offdiag_scale < 1:
        raise ValueError("Invalid keep fraction or off-diagonal shrinkage")
    if args.featcal_lambda < 0 or not 0 <= args.featcal_alpha <= 1:
        raise ValueError("Invalid FeatCal regularization or teacher interpolation")
    settings = {
        k: getattr(args, k)
        for k in (
            "method",
            "regmeanpp_solver",
            "offdiag_scale",
            "featcal_lambda",
            "featcal_rho",
            "featcal_alpha",
        )
    }
    settings.update(
        ridge_ratio=cfg["ridge_ratio"], max_correction_ratio=cfg["max_correction_ratio"]
    )
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    method=args.method,
                    output=str(output),
                    settings=settings,
                    calibration="none" if args.method in ("soup", "ties") else "cache_a; one pass",
                    ties_keep_fraction=args.ties_keep_fraction,
                    ties_alpha=args.ties_alpha,
                    dry_run=True,
                ),
                indent=2,
            )
        )
        return
    if args.method in ("soup", "ties"):
        report = parameter_merge(
            cfg, args.method, output, keep_fraction=args.ties_keep_fraction, alpha=args.ties_alpha
        )
    else:
        from tcr_merging.baselines.replay import ReplayBaseline
        from tcr_merging.merging.engine import merge_pass

        cfg.update(passes=1, variant="full", first_pass_weighting="none", output=str(output))
        cfg.pop("ridge_reference", None)
        if args.method == "featcal" and args.phase is None:
            output.parent.mkdir(parents=True, exist_ok=True)
            features = Path(tempfile.mkdtemp(prefix=".featcal-features-", dir=output.parent))
            common = [
                sys.executable,
                "-m",
                "tcr_merging.baselines.runner",
                "--config",
                str(args.config.resolve()),
                "--method",
                "featcal",
                "--output",
                str(output),
                "--features",
                str(features),
                "--featcal-lambda",
                str(args.featcal_lambda),
                "--featcal-rho",
                str(args.featcal_rho),
                "--featcal-alpha",
                str(args.featcal_alpha),
            ]
            print(f"FeatCal teacher-feature directory (retained for audit): {features}", flush=True)
            for phase in ("teacher", "student"):
                subprocess.run([*common, "--phase", phase], check=True)
            return
        if args.method == "featcal" and args.features is None:
            raise ValueError("Internal FeatCal phase requires a feature directory")
        backend = ReplayBaseline(
            cfg, output, settings, features=args.features, teacher_only=args.phase == "teacher"
        )
        report = merge_pass(cfg, 1, backend=backend)
    print(
        json.dumps(
            {
                "method": args.method,
                "output": str(output),
                "model_sha256": report.get("model_sha256"),
                "phase": args.phase,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
