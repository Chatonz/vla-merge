"""CPU end-to-end traversal on a tiny graph with all 418 production module names.

This validates orchestration/serialization, not real pi0.5 robot performance.
Native Gemma parity remains covered separately in test_native_replay.py.
"""

import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from tcr_merging.baselines.regmeanpp import solve_original_regmeanpp_weight
from tcr_merging.baselines.replay import ReplayBaseline
from tcr_merging.baselines.runner import main, parameter_merge
from tcr_merging.config import load_config
from tcr_merging.merging import engine
from tcr_merging.pi05.checkpoints import SUPPORT, sha256
from tcr_merging.pi05.replay import ReplayState, VisionReplayState
from tcr_merging.pi05.scope import target_keys
from torch import nn


class Block(nn.Module):
    def __init__(self, vision=False):
        super().__init__()
        self.vision = vision
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj" if vision else "o_proj"):
            setattr(self.self_attn, name, nn.Linear(2, 2, bias=False))
        self.mlp = nn.Module()
        for name in ("fc1", "fc2") if vision else ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp, name, nn.Linear(2, 2, bias=False))

    def forward(self, x, attention_mask=None, **kwargs):
        a = self.self_attn
        attn = a.q_proj(x) + a.k_proj(x) + a.v_proj(x)
        x = x + getattr(a, "out_proj" if self.vision else "o_proj")(attn) * 0.1
        m = self.mlp
        hidden = (
            torch.tanh(m.fc1(x)) if self.vision else torch.sigmoid(m.gate_proj(x)) * m.up_proj(x)
        )
        return x + getattr(m, "fc2" if self.vision else "down_proj")(hidden) * 0.1


class Norm(nn.Module):
    def forward(self, x, cond):
        return x, None


def decoder():
    model = nn.Module()
    model.layers = nn.ModuleList([Block() for _ in range(18)])
    model.config = SimpleNamespace()
    model.rotary_emb = lambda *args: None
    model.norm = Norm()
    return model


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        c = self.model
        c.config = SimpleNamespace(chunk_size=2)
        for name in ("action_in_proj", "time_mlp_in", "time_mlp_out", "action_out_proj"):
            setattr(c, name, nn.Linear(2, 2))
        c.paligemma_with_expert = nn.Module()
        p = c.paligemma_with_expert
        p.gemma_expert = nn.Module()
        p.gemma_expert.model = decoder()
        p.paligemma = nn.Module()
        p.paligemma.model = nn.Module()
        p.paligemma.model.language_model = decoder()
        p.paligemma.model.multi_modal_projector = nn.Identity()
        p.paligemma.model.vision_tower = nn.Module()
        v = nn.Module()
        p.paligemma.model.vision_tower.vision_model = v
        v.encoder = nn.Module()
        v.encoder.layers = nn.ModuleList([Block(True) for _ in range(27)])
        v.post_layernorm = nn.Identity()
        self.register_buffer("unchanged", torch.ones(1))

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = cls()
        model.load_state_dict(load_file(path / "model.safetensors"), strict=True)
        return model


@pytest.fixture
def bank(tmp_path, monkeypatch):
    torch.manual_seed(37)
    policy = Policy()
    base = policy.state_dict()
    names = ("task1", "task2", "task3")
    for i, name in enumerate(("base", *names)):
        root = tmp_path / name
        root.mkdir()
        (root / "config.json").write_text("{}")
        (root / "tokenizer").mkdir()
        (root / "tokenizer/tokenizer.json").write_text("{}")
        for sidecar in SUPPORT:
            if sidecar.endswith(".json"):
                (root / sidecar).write_text("{}")
            else:
                save_file({"scale": torch.ones(1)}, root / sidecar)
        tensors = {k: v.clone() + (0.005 * i if k in target_keys() else 0) for k, v in base.items()}
        save_file(tensors, root / "model.safetensors")
    manifest = {"samples": [{"flow_index": f, "selected_request_slot": 0} for f in (0, 5, 9)]}
    entries = {}
    for name in names:
        cache = tmp_path / f"cache-{name}"
        cache.mkdir()
        save_file({"dummy": torch.ones(1)}, cache / "replay.safetensors")
        (cache / "replay.json").write_text(json.dumps(manifest))
        cache_b = tmp_path / f"b-{name}"
        cache_b.mkdir()
        save_file({"dummy": torch.ones(1)}, cache_b / "replay.safetensors")
        (cache_b / "replay.json").write_text(json.dumps(manifest))
        entries[name] = {
            "checkpoint": str(tmp_path / name),
            "cache_a": str(cache),
            "cache_b": str(tmp_path / f"b-{name}"),
        }
    raw = {
        "base_model": str(tmp_path / "base"),
        "output": str(tmp_path / "tcr"),
        "experts": entries,
        "strict_paper_budget": False,
        "device": "cpu",
        "passes": 1,
        "rows_per_call": 2,
        "rows_per_request": 2,
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw))
    cfg = load_config(config)
    module = ModuleType("lerobot.policies.pi05.modeling_pi05")
    module.PI05Policy = Policy
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(engine, "validate_inputs", lambda cfg: None)

    def states(*args, **kwargs):
        torch.manual_seed(44)
        s = ReplayState(
            torch.randn(1, 2, 2),
            torch.randn(1, 2),
            torch.randn(1, 2, 2),
            torch.randn(1, 2),
            torch.zeros(1, 1, 2, 9),
            torch.tensor([[7, 8]]),
            [torch.zeros(1, 1, 7, 2) for _ in range(18)],
            [torch.zeros(1, 1, 7, 2) for _ in range(18)],
            vision=[VisionReplayState(torch.randn(1, 2, 2), None) for _ in range(3)],
            language_hidden_reference=torch.randn(1, 7, 2),
            language_attention_mask=torch.zeros(1, 1, 7, 7),
            language_position_ids=torch.arange(7)[None],
        )
        return [copy.deepcopy(s) for _ in range(3)], manifest

    monkeypatch.setattr(engine, "load_replay_states", states)
    monkeypatch.setattr(
        engine, "manual_action_block", lambda layer, rotary, hidden, *args: layer(hidden)
    )
    monkeypatch.setattr(
        engine,
        "decoder_prefix_kv",
        lambda layer, rotary, hidden, *args: (
            layer.self_attn.k_proj(hidden)[:, None],
            layer.self_attn.v_proj(hidden)[:, None],
        ),
    )
    return cfg, config


@pytest.mark.parametrize("method", ["soup", "ties"])
def test_parameter_export(bank, tmp_path, method):
    cfg, _ = bank
    report = parameter_merge(cfg, method, tmp_path / method, keep_fraction=1.0)
    assert report["modified_tensor_count"] == 422
    actual = load_file(tmp_path / method / "model.safetensors")
    base = load_file(tmp_path / "base/model.safetensors")
    for key in target_keys():
        torch.testing.assert_close(actual[key], base[key] + 0.01, atol=1e-7, rtol=1e-6)
    assert torch.equal(actual["unchanged"], base["unchanged"])
    with pytest.raises(FileExistsError):
        parameter_merge(cfg, method, tmp_path / method)


def test_parameter_rejects_out_of_scope(bank, tmp_path):
    cfg, _ = bank
    path = tmp_path / "task2/model.safetensors"
    tensors = load_file(path)
    tensors["unchanged"] *= 2
    save_file(tensors, path)
    with pytest.raises(ValueError, match="outside supported"):
        parameter_merge(cfg, "soup", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "method,solver",
    [
        ("regmeanpp", "original"),
        ("regmeanpp", "adapted"),
        ("featcal", "adapted"),
        ("tcr", "adapted"),
    ],
)
def test_all_418_modules_traverse_and_export(bank, tmp_path, method, solver):
    cfg, _ = bank
    settings = dict(
        method=method,
        regmeanpp_solver=solver,
        offdiag_scale=0.95,
        featcal_lambda=0.05,
        featcal_rho=2.0,
        featcal_alpha=0.3,
        ridge_ratio=0.05,
        max_correction_ratio=3.0,
    )
    output = tmp_path / ("out-" + method)
    features = tmp_path / "features"
    if method == "featcal":
        features.mkdir()
        teacher = ReplayBaseline(cfg, output, settings, features=features, teacher_only=True)
        receipt = engine.merge_pass(cfg, 1, backend=teacher)
        assert receipt["module_count"] == 418 and not output.exists()
    backend = None if method == "tcr" else ReplayBaseline(cfg, output, settings, features=features)
    result = engine.merge_pass(cfg, 1, backend=backend)
    assert result["module_count"] == 418
    path = tmp_path / "tcr/pass1" if method == "tcr" else output
    assert result["model_sha256"] == sha256(path / "model.safetensors")
    assert all(torch.isfinite(t).all() for t in load_file(path / "model.safetensors").values())
    if method == "featcal":
        assert result["modules"]["model.time_mlp_out"]["bias"]["bias_solve_after_weight"]
        assert not (output / "tcr_manifest.json").exists()


def test_original_solver_matches_three_expert_gram_oracle():
    torch.manual_seed(12)
    xs = {n: torch.randn(5, 3) for n in ("a", "b", "c")}
    ws = {n: torch.randn(2, 3) for n in xs}
    prior = sum(ws.values()) / 3
    result, info = solve_original_regmeanpp_weight(xs, ws, prior, offdiag_scale=0.7)
    grams = []
    for x in xs.values():
        gram = x.double().T @ x.double()
        grams.append(0.7 * gram + 0.3 * torch.diag(gram.diag()))
    expected = torch.linalg.solve(
        sum(grams), sum(g @ w.double().T for g, w in zip(grams, ws.values()))
    ).T
    torch.testing.assert_close(result.double(), expected, atol=1e-6, rtol=1e-6)
    assert not info["prior_ridge"] and not info["correction_cap"]


def test_original_rowspace_solver_and_inactive_support():
    torch.manual_seed(73)
    xs = {n: torch.randn(1, 9) for n in ("a", "b", "c")}
    for x in xs.values():
        x[:, -1] = 0
    ws = {n: torch.randn(2, 9) for n in xs}
    prior = sum(ws.values()) / 3
    actual, info = solve_original_regmeanpp_weight(xs, ws, prior, offdiag_scale=0.8)
    grams = []
    for x in xs.values():
        g = x[:, :-1].double().T @ x[:, :-1].double()
        grams.append(0.8 * g + 0.2 * torch.diag(g.diag()))
    expected = torch.linalg.solve(
        sum(grams), sum(g @ w[:, :-1].double().T for g, w in zip(grams, ws.values()))
    ).T
    torch.testing.assert_close(actual[:, :-1].double(), expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[:, -1], prior[:, -1])
    assert info["rowspace_kernel_shape"] == [3, 3]


def test_featcal_dependency_order():
    names = [
        "x.mlp.down_proj",
        "x.mlp.up_proj",
        "x.self_attn.o_proj",
        "x.self_attn.q_proj",
        "x.mlp.gate_proj",
    ]
    ordered = ReplayBaseline.order_groups({"action": {0: names}})["action"][0]
    assert ordered.index("x.self_attn.q_proj") < ordered.index("x.self_attn.o_proj")
    assert ordered.index("x.mlp.up_proj") < ordered.index("x.mlp.down_proj")
    assert ordered.index("x.mlp.gate_proj") < ordered.index("x.mlp.down_proj")


def test_tcr_two_pass_ridge_inheritance_unchanged(bank, tmp_path):
    cfg, _ = bank
    cfg["passes"] = 2
    first = engine.merge_pass(cfg, 1)
    second = engine.merge_pass(cfg, 2)
    assert second["method"] == "TCR"
    assert second["prior_model_sha256"] == first["model_sha256"]
    assert second["module_count"] == first["module_count"] == 418
    assert {n: v["ridge"] for n, v in first["modules"].items()} == {
        n: v["ridge"] for n, v in second["modules"].items()
    }


def test_baseline_refuses_ablation_config(bank, tmp_path):
    _, config = bank
    cfg = json.loads(config.read_text())
    cfg.update(variant="expert_prefix", ridge_reference="other")
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="not a TCR ablation"):
        main(
            [
                "--method",
                "regmeanpp",
                "--config",
                str(config),
                "--output",
                str(tmp_path / "bad"),
                "--dry-run",
            ]
        )


@pytest.mark.parametrize("method", ["soup", "ties", "regmeanpp", "featcal"])
def test_baseline_dry_run_no_outputs(bank, tmp_path, capsys, method):
    _, config = bank
    main(
        [
            "--method",
            method,
            "--config",
            str(config),
            "--output",
            str(tmp_path / "dry"),
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] and not (tmp_path / "dry").exists()
