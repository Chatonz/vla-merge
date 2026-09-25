"""Real LeRobot/Transformers layers at tiny dimensions, without model downloads."""

import pytest
import torch
from tcr_merging.pi05.replay import decoder_prefix_kv, manual_action_block

pi_gemma = pytest.importorskip("lerobot.policies.pi_gemma")
gemma = pytest.importorskip("transformers.models.gemma.modeling_gemma")
cache_utils = pytest.importorskip("transformers.cache_utils")


def layer_and_rotary(conditional):
    cfg = gemma.GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        vocab_size=32,
        max_position_embeddings=32,
    )
    cfg._attn_implementation = "eager"
    cfg.use_adarms = conditional
    cfg.adarms_cond_dim = 16 if conditional else None
    return pi_gemma._get_pi_gemma_decoder_layer_base()(cfg, 0).eval(), gemma.GemmaRotaryEmbedding(
        cfg
    )


@pytest.mark.parametrize("conditional", [False, True])
def test_manual_decoder_matches_native_with_prefix_cache(conditional):
    torch.manual_seed(17)
    layer, rotary = layer_and_rotary(conditional)
    hidden = torch.randn(1, 2, 16)
    cond = torch.randn(1, 16) if conditional else None
    key, value = torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4)
    positions = torch.tensor([[3, 4]])
    mask = torch.zeros(1, 1, 2, 5)
    cache = cache_utils.DynamicCache(((key.clone(), value.clone(), None),))
    with torch.inference_mode():
        actual = manual_action_block(layer, rotary, hidden, mask, positions, key, value, cond)
        native = layer(
            hidden,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=False,
            cache_position=torch.arange(3, 5),
            position_embeddings=rotary(hidden, positions),
            adarms_cond=cond,
        )
    torch.testing.assert_close(actual, native, atol=1e-6, rtol=1e-6)


def test_prefix_kv_matches_native_cache():
    torch.manual_seed(23)
    layer, rotary = layer_and_rotary(False)
    hidden = torch.randn(1, 3, 16)
    positions = torch.tensor([[0, 1, 2]])
    cache = cache_utils.DynamicCache()
    with torch.inference_mode():
        key, value = decoder_prefix_kv(layer, rotary, hidden, positions, None)
        layer(
            hidden,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(3),
            position_embeddings=rotary(hidden, positions),
        )
    torch.testing.assert_close(key, cache.layers[0].keys)
    torch.testing.assert_close(value, cache.layers[0].values)
