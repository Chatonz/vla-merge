"""The paper's pi0.5 scope: 418 linear modules and 422 tensors."""

VISION = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers."
LANGUAGE = "model.paligemma_with_expert.paligemma.model.language_model.layers."
ACTION = "model.paligemma_with_expert.gemma_expert.model.layers."
VISION_SUFFIXES = (
    "self_attn.k_proj",
    "self_attn.out_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
    "mlp.fc1",
    "mlp.fc2",
)
ACTION_LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
FRONTEND_MODULES = ("action_in_proj", "time_mlp_in", "time_mlp_out")
OUTPUT_MODULE = "action_out_proj"


def module_groups():
    return {
        "vision": {i: sorted(f"{VISION}{i}.{s}" for s in VISION_SUFFIXES) for i in range(27)},
        "language": {
            i: sorted(f"{LANGUAGE}{i}.{s}" for s in ACTION_LINEAR_SUFFIXES) for i in range(18)
        },
        "action": {i: [f"{ACTION}{i}.{s}" for s in ACTION_LINEAR_SUFFIXES] for i in range(18)},
    }


def target_keys():
    keys = {
        f"{name}.weight"
        for group in module_groups().values()
        for block in group.values()
        for name in block
    }
    keys.update(
        f"model.{name}.{kind}"
        for name in (*FRONTEND_MODULES, OUTPUT_MODULE)
        for kind in ("weight", "bias")
    )
    return keys
