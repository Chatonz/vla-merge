"""Small LIBERO entry point; the same collector can be used with other rollout code."""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

from tcr_merging.calibration.collection import CaptureConfig, ReplayCollector
from tcr_merging.pi05.runtime import visual_feature_compatibility


@contextmanager
def runtime_compatibility(init_state_offset=0):
    from lerobot.envs import libero

    original_init = libero.LiberoEnv.__init__

    def init(env, *args, **kwargs):
        original_init(env, *args, **kwargs)
        if env.init_states and env._init_states is not None:
            if not 0 <= init_state_offset < len(env._init_states):
                raise ValueError("Initial-state offset is outside the available bank")
            env.init_state_id += init_state_offset

    libero.LiberoEnv.__init__ = init
    try:
        with visual_feature_compatibility():
            yield
    finally:
        libero.LiberoEnv.__init__ = original_init


def collect_main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--selection", choices=("reservoir", "quantiles"), required=True)
    parser.add_argument("--noise-seed", type=int, default=272001)
    parser.add_argument("--seed", type=int, default=271001)
    parser.add_argument("--init-state-offset", type=int, required=True)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.scripts import lerobot_eval
    from tcr_merging.pi05.evaluation import eval_arguments

    config = CaptureConfig(
        args.name,
        args.policy,
        args.output,
        args.selection,
        args.noise_seed,
        args.seed,
        args.init_state_offset,
        args.expected_tasks,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    collector = None
    make_policy = lerobot_eval.make_policy
    prepare = Pi05PrepareStateTokenizerProcessorStep.__call__

    def make(*a, **kw):
        nonlocal collector
        policy = make_policy(*a, **kw)
        collector = ReplayCollector(policy, config)
        collector.install()
        return policy

    def prepare_state(step, transition):
        if collector is not None:
            collector.capture_task_texts(
                transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(step.task_key)
            )
        return prepare(step, transition)

    previous_argv = sys.argv
    lerobot_eval.make_policy = make
    Pi05PrepareStateTokenizerProcessorStep.__call__ = prepare_state
    sys.argv = [
        "lerobot-eval",
        *eval_arguments(
            args.policy, args.suite, args.output / "rollout", 1, args.seed, args.device
        ),
    ]
    try:
        with runtime_compatibility(args.init_state_offset):
            lerobot_eval.main()
        if collector is None:
            raise RuntimeError("No policy was constructed")
        collector.finish()
    finally:
        sys.argv = previous_argv
        lerobot_eval.make_policy = make_policy
        Pi05PrepareStateTokenizerProcessorStep.__call__ = prepare
        if collector is not None:
            collector.close()
