"""Run each (prompt length, mode) in a fresh process; preserve partial results."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIELDS = [
    "status", "mode", "prompt_tokens", "output_tokens", "repeat", "seed",
    "cuda_graphs", "graph_capacity", "graph_replays", "prompt_sha256",
    "ttft_ms", "mean_itl_ms", "p50_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "elapsed_s", "output_tokens_per_s", "decode_tokens_per_s",
    "peak_allocated_bytes", "peak_reserved_bytes", "allocated_bytes", "reserved_bytes",
    "kv_pool_bytes", "logical_kv_bytes_at_last_step", "processed_token_positions",
    "error",
]


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def processed_positions(prompt, output, cache, graph_capacity=None):
    # Algorithmic work proxy, NOT FLOPs or hardware compute utilization.
    if cache:
        return prompt + output - 1
    if graph_capacity:
        return prompt + (output-1)*graph_capacity
    return output*prompt + output*(output-1)//2


def save_csv(out):
    rows = []
    for path in sorted((out / "runs").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    with (out / "raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def worker(config, out, length, mode):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from nanovllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; select an A100 Colab runtime")
    seed, output = config["seed"], config["output_tokens"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = str(Path(config["model"]).resolve())
    hf = AutoConfig.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    special = set(tokenizer.all_special_ids)
    valid_ids = torch.tensor([i for i in range(hf.vocab_size) if i not in special], dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prompt = valid_ids[torch.randint(len(valid_ids), (length,), generator=generator)].tolist()
    prompt_text = json.dumps(prompt, separators=(",", ":"))
    prompt_sha = hashlib.sha256(prompt_text.encode()).hexdigest()
    (out / "prompts" / f"{length}.json").write_text(prompt_text + "\n")
    max_len = length + output
    if max_len > hf.max_position_embeddings:
        raise ValueError("Prompt + output exceeds checkpoint context window")
    cache = mode == "kv"
    graphs = config["cuda_graphs"]
    block_size = 256
    blocks = math.ceil(max_len / block_size)
    engine = LLM(model, benchmark=True,
                 use_kv_cache=cache, enable_prefix_caching=False,
                 enforce_eager=not graphs,
                 max_num_seqs=1, max_model_len=max_len,
                 max_num_batch_tokens=max_len,
                 no_cache_graph_capacity=max_len,
                 num_kvcache_blocks=blocks if cache else 0,
                 gpu_memory_utilization=config["gpu_memory_utilization"])
    params = SamplingParams(temperature=0.0, max_tokens=output, ignore_eos=True)
    for _ in range(config["warmups"]):
        engine.generate([prompt], params)
    props = torch.cuda.get_device_properties(0)
    versions = {}
    for package in ("torch", "transformers", "flash-attn", "triton", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    metadata = dict(gpu=props.name, total_memory_bytes=props.total_memory,
                    capability=list(torch.cuda.get_device_capability()),
                    torch_cuda=torch.version.cuda, python=platform.python_version(),
                    versions=versions, model_config=hf.to_dict(), config=config,
                    model_path=model, checkpoint_revision=config.get("model_revision"),
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    try:
        metadata["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
    except (OSError, subprocess.CalledProcessError):
        metadata["nvidia_smi"] = "unavailable"
    (out / "runs" / f"{length}_{mode}_environment.json").write_text(json.dumps(metadata, indent=2))
    kv_pool = sum(layer.self_attn.attn.k_cache.numel()*layer.self_attn.attn.k_cache.element_size()
                  + layer.self_attn.attn.v_cache.numel()*layer.self_attn.attn.v_cache.element_size()
                  for layer in engine.runner.model.layers)
    bytes_per_kv_token = 2*hf.num_hidden_layers*hf.num_key_value_heads*hf.head_dim*2
    raw_path = out / "runs" / f"{length}_{mode}.jsonl"
    token_path = out / "runs" / f"{length}_{mode}_tokens.csv"
    with raw_path.open("w") as raw, token_path.open("w", newline="") as tf:
        tw = csv.DictWriter(tf, fieldnames=["repeat", "request_id", "output_index", "context_tokens", "elapsed_s", "itl_s"])
        tw.writeheader()
        for repeat in range(config["repeats"]):
            engine.generate([prompt], params)
            m = engine.last_metrics
            if m["output_tokens"] != output:
                raise AssertionError("Benchmark did not produce the requested number of tokens")
            expected_replays = output-1 if graphs else 0
            if m["graph_replays"] != expected_replays:
                raise AssertionError(f"Expected {expected_replays} graph replays, got {m['graph_replays']}")
            itls = [e["itl_s"] for e in m["token_events"] if e["itl_s"] is not None]
            row = dict(status="ok", mode=mode, prompt_tokens=length, output_tokens=output,
                       repeat=repeat, seed=seed, cuda_graphs=graphs,
                       graph_capacity=max_len if graphs and not cache else 1 if graphs else 0,
                       graph_replays=m["graph_replays"], prompt_sha256=prompt_sha,
                       ttft_ms=m["ttft_s"]*1000, mean_itl_ms=m["mean_itl_s"]*1000,
                       p50_itl_ms=percentile(itls, .50)*1000,
                       p95_itl_ms=percentile(itls, .95)*1000,
                       p99_itl_ms=percentile(itls, .99)*1000,
                       elapsed_s=m["elapsed_s"], output_tokens_per_s=m["output_tokens_per_s"],
                       decode_tokens_per_s=(output-1)/sum(itls),
                       peak_allocated_bytes=m["peak_allocated_bytes"],
                       peak_reserved_bytes=m["peak_reserved_bytes"],
                       allocated_bytes=m["allocated_bytes"], reserved_bytes=m["reserved_bytes"],
                       kv_pool_bytes=kv_pool,
                       logical_kv_bytes_at_last_step=bytes_per_kv_token*(max_len-1) if cache else 0,
                       processed_token_positions=processed_positions(length, output, cache, max_len if graphs else None))
            raw.write(json.dumps(row)+"\n")
            raw.flush()
            for event in m["token_events"]:
                tw.writerow(dict(repeat=repeat, **event))
            tf.flush()
            (out / "runs" / f"{length}_{mode}_output_{repeat}.json").write_text(json.dumps(engine.output_token_ids[0]))
            print(json.dumps(row), flush=True)


def validate_config(config):
    for key in ("output_tokens", "repeats", "warmups"):
        if not isinstance(config[key], int) or config[key] < (2 if key == "output_tokens" else 1):
            raise ValueError(f"Invalid {key}")
    lengths = config["prompt_lengths"]
    if not lengths or len(set(lengths)) != len(lengths) or any(not isinstance(n, int) or n < 1 for n in lengths):
        raise ValueError("Prompt lengths must be unique positive integers")
    if not isinstance(config["cuda_graphs"], bool):
        raise ValueError("cuda_graphs must be boolean")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmarks/kv_cache/a100.json")
    parser.add_argument("--out", default="results/kv_cache_a100")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["kv", "no_kv"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path, out = Path(args.config).resolve(), Path(args.out).resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    if args.worker:
        worker(config, out, args.length, args.mode)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}; choose a new --out")
    for subdir in ("runs", "prompts", "logs"):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    failures = []
    for i, length in enumerate(config["prompt_lengths"]):
        modes = ("kv", "no_kv") if i % 2 == 0 else ("no_kv", "kv")
        for mode in modes:
            print(f"Running prompt={length} mode={mode} (see {out / 'logs'})", flush=True)
            log = out / "logs" / f"{length}_{mode}.log"
            with log.open("w") as f:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--config", str(config_path), "--out", str(out), "--worker",
                    "--length", str(length), "--mode", mode], stdout=f, stderr=subprocess.STDOUT)
            if result.returncode:
                failures.append(f"{length}/{mode}")
                with (out / "runs" / f"{length}_{mode}.jsonl").open("a") as f:
                    f.write(json.dumps(dict(status="error", mode=mode, prompt_tokens=length,
                                            error=f"Worker exit {result.returncode}; see {log.name}"))+"\n")
                print(f"FAILED {length}/{mode}: inspect {log}", flush=True)
            save_csv(out)
    subprocess.run([sys.executable, str(Path(__file__).with_name("plot.py")), "--results", str(out)], check=True)
    if failures:
        raise SystemExit("Some cases failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
