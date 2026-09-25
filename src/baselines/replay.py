"""Baseline solvers plugged into the shared native pi0.5 traversal.

FeatCal first records unmodified expert features, then replays the student in
dependency order, refreshing producers after each calibrated linear module.
Neither feature statistics nor teacher targets come from TCR's local loss.
"""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tcr_merging.baselines.featcal import (
    FeatCalConfig,
    featcal_linear_bias,
    featcal_linear_weight,
    linear_feature_statistics,
)
from tcr_merging.baselines.regmeanpp import solve_original_regmeanpp_weight
from tcr_merging.merging.regression import solve_weight_multi
from tcr_merging.pi05.checkpoints import sha256


class ReplayBaseline:
    def __init__(self, cfg, output, settings, *, features=None, teacher_only=False):
        self.cfg = cfg
        self.output = Path(output)
        self.settings = settings
        self.method = settings["method"]
        self.teacher_only = teacher_only
        self.student_inputs = self.method == "featcal" and not teacher_only
        self.features = Path(features) if features is not None else None
        self.feature_files = {}
        self.description = (
            "FeatCal full-expert teacher / sequential student; weight and bias equations"
            if self.method == "featcal"
            else "Original RegMean++ raw Gram shrinkage; mean interface biases; no ridge/cap"
            if settings["regmeanpp_solver"] == "original"
            else "Adapted RegMean++: one pass, equal masses, merged prefix, Soup ridge/cap"
        )
        if self.student_inputs:
            receipt = json.loads((self.features / "teacher_manifest.json").read_text())
            if receipt["config"] != cfg or receipt["module_count"] != 418:
                raise ValueError("FeatCal teacher features do not match this configuration")
            for name, entry in cfg["experts"].items():
                if (
                    sha256(Path(entry["checkpoint"]) / "model.safetensors")
                    != receipt["expert_sha256"][name]
                ):
                    raise ValueError("FeatCal teacher weights changed")
                if (
                    sha256(Path(entry["cache_a"]) / "replay.safetensors")
                    != receipt["cache_sha256"][name]
                ):
                    raise ValueError("FeatCal input cache changed")
                if (
                    sha256(Path(entry["cache_a"]) / "replay.json")
                    != receipt["cache_manifest_sha256"][name]
                ):
                    raise ValueError("FeatCal input metadata changed")
            self.feature_files = receipt["files"]

    @staticmethod
    def order_groups(groups):
        # A calibrated producer must be visible to every subsequent consumer.
        def rank(name):
            suffixes = (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "self_attn.o_proj",
                "mlp.fc1",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.fc2",
                "mlp.down_proj",
            )
            return next(i for i, suffix in enumerate(suffixes) if name.endswith(suffix))

        return {
            group: {i: sorted(names, key=rank) for i, names in blocks.items()}
            for group, blocks in groups.items()
        }

    def solve(self, module, inputs, weights, prior, *, bias=False):
        names = list(inputs)
        if set(names) != set(self.cfg["experts"]):
            raise ValueError("Baseline inputs must contain exactly the configured experts")
        rows = {n: len(inputs[n]) for n in names}
        if len(set(rows.values())) != 1:
            raise ValueError("Matched baseline requires equal calibration rows per expert")
        weights = {n: w.to(prior.device) for n, w in weights.items()}
        inputs = {n: x.to(device=prior.device, dtype=torch.float32) for n, x in inputs.items()}
        if self.teacher_only:
            # Interfaces in the shared traversal use augmented inputs. FeatCal
            # solves their weight and bias separately, so omit the constant here.
            path = self.features / (hashlib.sha256(module.encode()).hexdigest() + ".safetensors")
            if path.exists():
                raise FileExistsError(path)
            save_file(
                {
                    n: (x[:, :-1] if bias else x).detach().cpu().contiguous()
                    for n, x in inputs.items()
                },
                path,
            )
            self.feature_files[module] = {"file": path.name, "sha256": sha256(path)}
            return prior.detach().cpu(), {
                "rows_by_expert": rows,
                "improvement_vs_prior": 0.0,
                "operation": "record_original_teacher_inputs",
            }
        if self.method == "regmeanpp":
            if self.settings["regmeanpp_solver"] == "adapted":
                merged, info = solve_weight_multi(
                    inputs,
                    weights,
                    prior,
                    self.settings["ridge_ratio"],
                    "feature_energy",
                    self.settings["max_correction_ratio"],
                    "none",
                )
                info["recipe"] = "adapted_regmeanpp_one_pass_uniform"
                return merged, info
            x = {n: v[:, :-1] if bias else v for n, v in inputs.items()}
            w = {n: v[:, :-1] if bias else v for n, v in weights.items()}
            reference = prior[:, :-1] if bias else prior
            merged, info = solve_original_regmeanpp_weight(
                x,
                w,
                reference,
                offdiag_scale=self.settings["offdiag_scale"],
            )
            if bias:
                mean_bias = sum(v[:, -1] for v in weights.values()) / len(names)
                merged = torch.cat((merged, mean_bias.cpu()[:, None]), dim=1)
                info["bias_semantics"] = "uniform_expert_mean_not_augmented_regression"
            return merged, info

        record = self.feature_files[module]
        path = self.features / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError("FeatCal teacher feature hash mismatch")
        teacher = {n: x.float() for n, x in load_file(path, device=str(prior.device)).items()}
        if set(teacher) != set(names):
            raise ValueError("FeatCal teacher identities differ")
        x = {n: v[:, :-1] if bias else v for n, v in inputs.items()}
        ws = [weights[n][:, :-1] if bias else weights[n] for n in names]
        config = FeatCalConfig(
            ridge_lambda=self.settings["featcal_lambda"],
            anchor_blend_rho=self.settings["featcal_rho"],
            teacher_interp_alpha=self.settings["featcal_alpha"],
        )
        stats = [
            linear_feature_statistics(x[n], teacher[n], alpha=config.teacher_interp_alpha)
            for n in names
        ]
        with safe_open(
            Path(self.cfg["base_model"]) / "model.safetensors", framework="pt", device="cpu"
        ) as base:
            base_weight = base.get_tensor(module + ".weight").to(prior.device).float()
            base_bias = base.get_tensor(module + ".bias").to(prior.device).float() if bias else None
        soup_weight = prior[:, :-1] if bias else prior
        merged, info = featcal_linear_weight(
            ws,
            stats,
            soup_weight=soup_weight,
            base_weight=base_weight,
            config=config,
        )
        if bias:
            b, bias_info = featcal_linear_bias(
                [weights[n][:, -1] for n in names],
                ws,
                stats,
                calibrated_weight=merged,
                soup_bias=prior[:, -1],
                base_bias=base_bias,
                config=config,
            )
            merged = torch.cat((merged, b[:, None]), dim=1)
            info["bias"] = bias_info
        before = sum(
            float(((inputs[n] @ (prior - weights[n].to(prior.device)).T) ** 2).mean())
            for n in names
        )
        after = sum(
            float(((inputs[n] @ (merged - weights[n].to(prior.device)).T) ** 2).mean())
            for n in names
        )
        info.update(
            rows_by_expert=rows,
            improvement_vs_prior=(before - after) / max(before, 1e-12),
            teacher_features="original_expert_full_prefix",
            student_features="refreshed_calibrated_prefix",
        )
        return merged.detach().cpu(), info

    def finish_teacher(self, expert_hashes, cache_hashes, replay_error):
        if len(self.feature_files) != 418:
            raise ValueError("Incomplete FeatCal teacher feature collection")
        report = dict(
            config=self.cfg,
            module_count=len(self.feature_files),
            expert_sha256=expert_hashes,
            cache_sha256=cache_hashes,
            cache_manifest_sha256={
                n: sha256(Path(e["cache_a"]) / "replay.json")
                for n, e in self.cfg["experts"].items()
            },
            files=self.feature_files,
            manual_native_block_max_error=replay_error,
        )
        with (self.features / "teacher_manifest.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        return report
