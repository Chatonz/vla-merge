"""Reusable frozen-expert capture for execution and demonstration observations."""

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from tcr_merging.calibration._collector import BlockCalibrationCollector
from tcr_merging.pi05.checkpoints import sha256


@dataclass
class CaptureConfig:
    name: str
    checkpoint: Path
    output: Path
    selection: str = "reservoir"
    noise_seed: int = 272001
    simulator_seed: int = 271001
    init_state_offset: int = 0
    expected_tasks: int = 10
    max_requests: int = 128


@contextmanager
def environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def select_quantiles(metadata, count=5):
    requests = sorted({s["request_index"] for s in metadata})
    if len(requests) < count:
        raise ValueError("Episode has fewer than five requests; do not pad or select by success")
    chosen = [requests[(len(requests) - 1) * i // (count - 1)] for i in range(count)]
    indices = []
    for request in chosen:
        group = sorted(
            (i for i, s in enumerate(metadata) if s["request_index"] == request),
            key=lambda i: metadata[i]["flow_index"],
        )
        if [metadata[i]["flow_index"] for i in group] != [0, 5, 9]:
            raise ValueError("Incomplete generation path")
        indices.extend(group)
    return indices


class ReplayCollector(BlockCalibrationCollector):
    """Install once on a dense policy; call finish only after all episodes complete.

    Call policy.reset() per episode and policy.predict_action_chunk() normally.
    Inputs must be preprocessed exactly as at deployment. Batch size is one.
    """

    def __init__(self, policy, config: CaptureConfig):
        if config.selection not in ("reservoir", "quantiles", "initial"):
            raise ValueError("selection must be reservoir, quantiles or initial")
        config.checkpoint = Path(config.checkpoint).resolve()
        config.output = Path(config.output).resolve()
        if any((config.output / name).exists() for name in ("replay.json", "replay.safetensors")):
            raise FileExistsError(config.output)
        if getattr(policy.config, "use_peft", False):
            raise ValueError("Capture requires a materialized dense expert")
        if policy.config.num_inference_steps != 10:
            raise ValueError("This recipe expects ten native generation calls")
        settings = dict(
            TASK=config.name,
            CALIBRATION_POLICY=config.checkpoint,
            TENSOR_OUTPUT=config.output / "replay.safetensors",
            MANIFEST_OUTPUT=config.output / "replay.json",
            MAX_CALLS=150,
            MAX_CALLS_PER_PROMPT=1,
            REQUESTS_PER_EPISODE=config.max_requests if config.selection == "quantiles" else 5,
            EPISODE_AWARE=1,
            FULL_PREFIX=1,
            FLOW_INDICES="0,5,9",
            REQUEST_MODE="reservoir" if config.selection == "reservoir" else "initial",
            START_SEED=config.simulator_seed,
        )
        values = {"PI05_BLOCK_REGMEANPP_" + k: v for k, v in settings.items()}
        values["PI05_LIBERO_INIT_STATE_OFFSET"] = config.init_state_offset
        with environment(values):
            super().__init__(policy)
        self.capture_config = config
        self.original_noise = policy.model.sample_noise
        self.noise_hashes = []
        self.finished = False

    def install(self):
        self.policy.eval()
        self.policy.requires_grad_(False)
        super().install()
        device = next(self.policy.parameters()).device
        generator = torch.Generator(device=device).manual_seed(self.capture_config.noise_seed)

        def noise(shape, device):
            value = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
            self.noise_hashes.append(
                hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            )
            return value

        self.policy.model.sample_noise = noise

    def close(self):
        self.policy.model.sample_noise = self.original_noise
        if self.original_predict_action_chunk is not None:
            self.policy.predict_action_chunk = self.original_predict_action_chunk
        if self.original_policy_reset is not None:
            self.policy.reset = self.original_policy_reset
        for hook in self.handles:
            hook.remove()

    def finish(self, *, source_kind="expert_execution", extra=None):
        if self.finished:
            raise RuntimeError("Capture already finalized")
        cfg = self.capture_config
        if len(self.records_by_prompt) != cfg.expected_tasks:
            raise ValueError(
                f"Expected {cfg.expected_tasks} tasks, captured {len(self.records_by_prompt)}"
            )
        for prompt, records in self.records_by_prompt.items():
            metadata = self.record_metadata_by_prompt[prompt]
            requests = sorted({m["request_index"] for m in metadata if m})
            if cfg.selection == "quantiles":
                if len(requests) >= cfg.max_requests:
                    raise ValueError(
                        "Episode reached capture cap; increase max_requests and recollect"
                    )
                indices = select_quantiles(metadata)
                self.records_by_prompt[prompt] = [records[i] for i in indices]
                self.record_metadata_by_prompt[prompt] = [metadata[i] for i in indices]
                metadata = self.record_metadata_by_prompt[prompt]
            requests = sorted({m["request_index"] for m in metadata if m})
            if (
                len(metadata) != 15
                or len(requests) != 5
                or any(not r for r in self.records_by_prompt[prompt])
            ):
                raise ValueError("Each task must supply five complete request/flow triples")
            for sample in metadata:
                sample["selected_request_slot"] = requests.index(sample["request_index"])
        self.requests_per_episode = 5
        super().save()
        manifest = json.loads(self.manifest_output.read_text())
        manifest.update(
            source_kind=source_kind,
            request_selection=cfg.selection,
            generation_noise_seed=cfg.noise_seed,
            native_noise_sha256=self.noise_hashes,
            model_sha256=sha256(cfg.checkpoint / "model.safetensors"),
            tensor_sha256=sha256(self.tensor_output),
            success_filter=False,
        )
        if extra:
            manifest.update(extra)
        self.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
        self.finished = True
        self.close()
        return manifest
