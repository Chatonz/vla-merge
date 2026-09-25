"""Replace execution observations with demonstration observations, retaining native generation."""

import argparse
import json
from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Local LeRobot dataset root")
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="JSON list: one prespecified episode index per task",
    )
    parser.add_argument(
        "--template", type=Path, required=True, help="Matching execution cache A or B"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-stride", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    episodes = json.loads(args.episodes.read_text())
    if not episodes or len(set(episodes)) != len(episodes) or args.request_stride < 1:
        raise ValueError("Provide distinct episodes and a positive stride")
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from safetensors import safe_open
    from tcr_merging.calibration.collection import CaptureConfig, ReplayCollector
    from tcr_merging.pi05.checkpoints import sha256
    from tcr_merging.pi05.runtime import visual_feature_compatibility

    template = json.loads((args.template / "replay.json").read_text())
    if Path(template["calibration_policy"]).resolve() != args.policy.resolve():
        raise ValueError("Noise template must come from this frozen expert")
    initial_calls = [s for s in template["samples"] if s["flow_index"] == 0]
    by_text = {}
    for sample in initial_calls:
        text = template["prompt_texts"][str(sample["prompt_signature"])]
        by_text.setdefault(text, []).append(sample)
    noise_indices = {}
    for text, group in by_text.items():
        if len(group) != 5:
            raise ValueError("Noise template must contain five requests per task")
        for slot, sample in enumerate(sorted(group, key=lambda s: s["request_index"])):
            noise_indices[text, slot] = sample["index"]
    dataset = LeRobotDataset(
        args.repo_id, root=args.dataset, episodes=episodes, video_backend="pyav"
    )
    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.pretrained_path = args.policy.resolve()
    cfg.device = args.device
    cfg.compile_model = False
    cfg.n_action_steps = 10
    with visual_feature_compatibility():
        policy = make_policy(cfg, ds_meta=dataset.meta).eval()
    pre, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    collector = ReplayCollector(
        policy,
        CaptureConfig(
            args.name, args.policy, args.output, selection="initial", expected_tasks=len(episodes)
        ),
    )
    rows = dataset.hf_dataset.select_columns(["episode_index", "frame_index"]).to_pandas()
    provenance = {}
    seen_tasks = set()
    collector.install()
    try:
        with safe_open(
            args.template / "replay.safetensors", framework="pt", device="cpu"
        ) as handle:
            for episode in episodes:
                candidates = list(
                    rows[rows.episode_index == episode].sort_values("frame_index").index
                )[:: args.request_stride]
                if len(candidates) < 5:
                    raise ValueError("Demonstration too short for five distinct observations")
                selected = [candidates[(len(candidates) - 1) * i // 4] for i in range(5)]
                policy.reset()
                for slot, index in enumerate(selected):
                    sample = dataset[int(index)]
                    text = sample["task"]
                    if slot == 0:
                        if text in seen_tasks:
                            raise ValueError("Select one demonstration episode per task")
                        seen_tasks.add(text)
                    noise = handle.get_tensor(
                        f"sample_{noise_indices[text, slot]:03d}.action_input"
                    )

                    def paired_noise(shape, device, value=noise):
                        if tuple(shape) != tuple(value.shape):
                            raise ValueError("Template noise shape differs from native generator")
                        return value.to(device).clone()

                    policy.model.sample_noise = paired_noise
                    observation = {k: v for k, v in sample.items() if k.startswith("observation.")}
                    observation["task"] = text
                    collector.capture_task_texts([text])
                    with torch.inference_mode():
                        policy.predict_action_chunk(pre(observation))
                    provenance[collector.latest_prompt_signature, slot] = dict(
                        demo_episode_index=episode,
                        demo_frame_index=int(rows.loc[index, "frame_index"]),
                    )
        for signature, metadata in collector.record_metadata_by_prompt.items():
            for sample in metadata:
                sample.update(provenance[signature, sample["request_index"]])
                sample.update(
                    init_state_id=None, simulator_seed=None, initial_observation_sha256=None
                )
        collector.finish(
            source_kind="demonstration_observations",
            extra={
                "demonstration_actions_used": False,
                "paired_noise_template": str(args.template.resolve()),
                "paired_noise_template_sha256": sha256(args.template / "replay.safetensors"),
                "demonstration_episodes": episodes,
                "generation_noise_seed": None,
                "dataset_root": str(args.dataset.resolve()),
                "request_stride": args.request_stride,
            },
        )
    finally:
        collector.close()
