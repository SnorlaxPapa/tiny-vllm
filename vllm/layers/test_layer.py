import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP as HFQwen2MLP, apply_rotary_pos_emb, rotate_half
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as HFQwen2ROPE
from .mlp_layer import GatedLinear
from .rope import RotaryEmbedding

def test_mlp():
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    reference = HFQwen2MLP(config).to("cuda")
    custom = GatedLinear(config).to("cuda")

    custom.load_state_dict(reference.state_dict())

    x = torch.randn(2, 8, config.hidden_size, device="cuda")
    expected = reference(x)
    actual = custom(x)

    torch.testing.assert_close(actual, expected)
    print("success")

def test_rope():
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct"
    )

    head_dim = (
        config.head_dim
        if hasattr(config, "head_dim")
        else config.hidden_size // config.num_attention_heads
    )

    reference = HFQwen2ROPE(config, device="cuda")

    custom = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_parameters["rope_theta"],
    ).to("cuda")

    num_tokens = 8

    positions = torch.arange(
        num_tokens,
        device="cuda",
    )

    query = torch.randn(
        num_tokens,
        config.num_attention_heads,
        head_dim,
        device="cuda",
    )

    key = torch.randn(
        num_tokens,
        config.num_key_value_heads,
        head_dim,
        device="cuda",
    )

    # HF expects (B, H, S, D).
    query_ref = query.transpose(0, 1).unsqueeze(0)
    key_ref = key.transpose(0, 1).unsqueeze(0)
    positions_ref = positions.unsqueeze(0)

    cos, sin = reference(query_ref, positions_ref)

    expected_query, expected_key = apply_rotary_pos_emb(
        query_ref,
        key_ref,
        cos,
        sin,
    )

    # Convert HF output back to (T, H, D).
    expected_query = expected_query.squeeze(0).transpose(0, 1)
    expected_key = expected_key.squeeze(0).transpose(0, 1)

    actual_query, actual_key = custom(
        positions,
        query,
        key,
    )

    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)

    print("match")


test_rope()