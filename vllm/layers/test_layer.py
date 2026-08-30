import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as HFQwen2MLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as HFQwen2ROPE
from .mlp_layer import Qwen2MLP
from .rope import Qwen2RotaryEmbedding

def test_mlp():
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    reference = HFQwen2MLP(config).to("cuda")
    custom = Qwen2MLP(config).to("cuda")

    custom.load_state_dict(reference.state_dict())

    x = torch.randn(2, 8, config.hidden_size, device="cuda")
    expected = reference(x)
    actual = custom(x)

    torch.testing.assert_close(actual, expected)
    print("success")

def test_rope():
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    reference = HFQwen2ROPE(config, device="cuda")
    custom = Qwen2RotaryEmbedding(config, device="cuda")

    x = torch.randn(2, 8, config.hidden_size, device="cuda")
    position_ids = torch.arange(8, device="cuda").unsqueeze(0).expand(2, -1)
    cos_ref, sin_ref = reference(x, position_ids)
    cos_actual, sin_actual = custom(x, position_ids)

    torch.testing.assert_close(cos_ref, cos_actual)
    torch.testing.assert_close(sin_ref, sin_actual)

    print("match")


test_rope()