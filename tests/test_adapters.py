import pytest
import torch
from tcr_merging.pi05.adapters import peft_cpu_safe_merge_lora_weight, validate_adapter_config


def test_bfloat16_delta_rounding_precedes_base_addition():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(8, 8, generator=generator).bfloat16()
    a = torch.randn(3, 8, generator=generator).bfloat16()
    b = torch.randn(8, 3, generator=generator).bfloat16()
    actual = peft_cpu_safe_merge_lora_weight(base, a, b, 0.7)
    expected = base + ((b.float() @ a.float()) * 0.7).to(base.dtype)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, (base.float() + (b.float() @ a.float()) * 0.7).to(base.dtype))


@pytest.mark.parametrize("option", ["use_dora", "use_rslora", "fan_in_fan_out"])
def test_unsupported_adapter_modes_rejected(option):
    with pytest.raises(ValueError):
        validate_adapter_config({"peft_type": "LORA", "r": 4, "lora_alpha": 8, option: True})
