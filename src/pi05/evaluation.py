"""Evaluate one checkpoint with explicit suite, reset range and seed."""

import argparse
import sys
from pathlib import Path


def eval_arguments(policy, suite, output, episodes, seed, device):
    return [
        f"--policy.path={Path(policy).resolve()}",
        f"--policy.device={device}",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=false",
        "--policy.n_action_steps=10",
        "--env.type=libero",
        f"--env.task={suite}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={episodes}",
        f"--seed={seed}",
        f"--output_dir={Path(output).resolve()}",
    ]


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per task")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--init-state-offset", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    from lerobot.scripts import lerobot_eval
    from tcr_merging.pi05.libero import runtime_compatibility

    previous = sys.argv
    sys.argv = [
        "lerobot-eval",
        *eval_arguments(
            args.policy, args.suite, args.output, args.episodes, args.seed, args.device
        ),
    ]
    try:
        with runtime_compatibility(args.init_state_offset):
            lerobot_eval.main()
    finally:
        sys.argv = previous
