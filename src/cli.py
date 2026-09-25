"""Command-line entry points with lazy model/runtime imports."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "baseline":
        from tcr_merging.baselines.runner import main as run_baseline

        return run_baseline(argv[1:])
    # Delegate specialized help without importing the simulator in ordinary commands.
    if argv and argv[0] in ("collect", "collect-demo", "evaluate"):
        if argv[0] == "collect":
            from tcr_merging.pi05.libero import collect_main as run
        elif argv[0] == "collect-demo":
            from tcr_merging.calibration.demo import main as run
        else:
            from tcr_merging.pi05.evaluation import main as run
        return run(argv[1:])
    parser = argparse.ArgumentParser(description="TCR-Merging: pi0.5 fusion and matched controls")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("merge", "merge-pass", "check"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name == "merge":
            sub.add_argument("--dry-run", action="store_true")
        if name == "merge-pass":
            sub.add_argument("--pass-index", type=int, choices=(1, 2), required=True)
    export = commands.add_parser("export-adapter", help="Materialize a PEFT LoRA expert")
    export.add_argument("--adapter", type=Path, required=True)
    export.add_argument("--base-model", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    sweep = commands.add_parser(
        "sweep", help="Write configurations and commands; does not launch jobs"
    )
    sweep.add_argument("--config", type=Path, required=True)
    sweep.add_argument(
        "--study", choices=("ablations", "ridge", "budget", "passes", "weighting"), required=True
    )
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--demo-config", type=Path)
    commands.add_parser("collect", help="Capture LIBERO expert execution")
    commands.add_parser(
        "collect-demo", help="Capture native generation from demonstration observations"
    )
    commands.add_parser("evaluate", help="Evaluate one checkpoint on LIBERO")
    commands.add_parser("baseline", help="Independent Soup, TIES, RegMean++ and FeatCal")
    args = parser.parse_args(argv)
    if args.command == "sweep":
        from tcr_merging.experiments import generate

        result = generate(args.config, args.study, args.output, args.demo_config)
    elif args.command == "export-adapter":
        from tcr_merging.pi05.adapters import materialize_checkpoint

        result = materialize_checkpoint(
            adapter_root=args.adapter, base_root=args.base_model, output=args.output
        )
    else:
        from tcr_merging.config import load_config, validate_inputs

        cfg = load_config(args.config)
        if args.command == "check":
            validate_inputs(cfg)
            result = {
                "inputs": "passed",
                "note": "Checks paths, support files and trace metadata; not a model forward test.",
            }
        elif args.command == "merge-pass":
            from tcr_merging.merging.engine import merge_pass

            merge_pass(cfg, args.pass_index)
            return
        else:
            commands_to_run = [
                [
                    sys.executable,
                    "-m",
                    "tcr_merging.cli",
                    "merge-pass",
                    "--config",
                    str(args.config.resolve()),
                    "--pass-index",
                    str(i),
                ]
                for i in range(1, cfg["passes"] + 1)
            ]
            if not args.dry_run:
                for name in ("pass1", "merged")[: cfg["passes"]]:
                    if (Path(cfg["output"]) / name).exists():
                        raise FileExistsError(Path(cfg["output"]) / name)
                validate_inputs(cfg)
                for command in commands_to_run:
                    subprocess.run(command, check=True)
            result = {"commands": commands_to_run, "output": cfg["output"], "dry_run": args.dry_run}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
