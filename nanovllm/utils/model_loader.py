import torch
import gc
from safetensors import safe_open
from transformers import AutoConfig
from nanovllm.layers.rope import RotaryEmbedding
from nanovllm.model.qwen2 import Qwen2Model


def initialize_rope(config, model):
  head_dim = (
    getattr(config, "head_dim", None)
    or config.hidden_size // config.num_attention_heads
  )

  rope = RotaryEmbedding(
      head_dim,
      head_dim,
      config.max_position_embeddings,
      config.rope_parameters['rope_theta']
  )

  rope = rope.to("cuda")

  for layer in model.layers:
    layer.self_attn.rotate = rope

  return model


def load_model(model_dir):
  """maps weights to parameter name and load"""
  tensors = {}

  for i in range(1, 3):
    with safe_open(f"{model_dir}/model-0000{i}-of-00002.safetensors", framework="pt", device=0) as f:
        for k in f.keys():
          tensors[k] = f.get_tensor(k)

  config = AutoConfig.from_pretrained(model_dir)

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

    else:
      tensor = tensors.pop(f"model.{name}")

    assert tensor.shape == param.shape, (
          f"{name}: checkpoint {tensor.shape}, model {param.shape}"
    )

    mapped_weights[name] = tensor


  model.load_state_dict(mapped_weights, assign=True, strict=True)
  #initialize global rope layer
  model = initialize_rope(config, model)

  model.eval()

  del mapped_weights
  del tensors
  gc.collect()
  torch.cuda.empty_cache()


  return model



