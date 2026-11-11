import torch
import gc
import json
from pathlib import Path
from safetensors import safe_open
from transformers import AutoConfig
from nanovllm.layers.rope import RotaryEmbedding, rope_theta
from nanovllm.model.qwen2 import Qwen2Model


def initialize_rope(config, model, device="cuda"):
  head_dim = (
    getattr(config, "head_dim", None)
    or config.hidden_size // config.num_attention_heads
  )

  rope = RotaryEmbedding(
      head_dim,
      head_dim,
      config.max_position_embeddings,
      rope_theta(config)
  )

  rope = rope.to(device)

  for layer in model.layers:
    layer.self_attn.rotate = rope

  return model


def load_model(model_dir, device="cuda", dtype=torch.bfloat16):
  """maps weights to parameter name and load"""
  tensors = {}

  model_path = Path(model_dir)
  index = model_path / "model.safetensors.index.json"
  if index.exists():
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
  else:
    shards = ["model.safetensors"]
  for shard in shards:
    with safe_open(str(model_path / shard), framework="pt", device="cpu") as f:
      for k in f.keys():
        tensors[k] = f.get_tensor(k)

  config = AutoConfig.from_pretrained(model_dir)
  if config.model_type not in ("qwen2", "qwen3"):
    raise ValueError(f"Unsupported architecture: {config.model_type}")
  if getattr(config, "use_sliding_window", False) or getattr(config, "rope_scaling", None):
    raise ValueError("Sliding-window attention and scaled RoPE are not implemented")

  with torch.device("meta"):
    model = Qwen2Model(config)

  mapped_weights = {}
  for name, param in model.named_parameters():

    if "qkv_proj.linear" in name: 
      prefix, kind = name.split("qkv_proj.linear.")

      q = tensors.pop(f"model.{prefix}q_proj.{kind}")
      k = tensors.pop(f"model.{prefix}k_proj.{kind}")
      v = tensors.pop(f"model.{prefix}v_proj.{kind}")
      tensor = torch.cat([q, k, v], dim=0)

      #free up space
      del q, k, v
    
    elif "mlp.gate_up_proj.weight" in name:
      prefix = name.split(".mlp")[0]
      up = tensors.pop(f"model.{prefix}.mlp.up_proj.weight")
      gate = tensors.pop(f"model.{prefix}.mlp.gate_proj.weight")

      tensor = torch.cat([gate, up], dim=0)

      del up, gate

    else:
      tensor = tensors.pop(name if name.startswith("lm_head.") else f"model.{name}")

    assert tensor.shape == param.shape, (
          f"{name}: checkpoint {tensor.shape}, model {param.shape}"
    )

    mapped_weights[name] = tensor.to(device=device, dtype=dtype)


  model.load_state_dict(mapped_weights, assign=True, strict=True)
  #initialize global rope layer
  model = initialize_rope(config, model, device=device)

  model.eval()

  del mapped_weights
  del tensors
  gc.collect()
  torch.cuda.empty_cache()


  return model



