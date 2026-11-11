# Qwen3-4B: KV cache on versus full-prefix recomputation

**Status: implementation and CPU tests provided; real A100 measurements are pending.** No timing CSV or performance plot in this PR is presented as measured GPU data. Running the commands below creates the raw CSVs, plots and a results README automatically. The older repository benchmark and `assets/benchmark.png` concern Qwen2.5 and are not evidence for this experiment.

[Open the Colab notebook](https://colab.research.google.com/github/SnorlaxPapa/tiny-vllm/blob/feat/qwen3-kv-cache-ablation/benchmarks/kv_cache/colab_a100.ipynb).

## Research question and controlled variables

How does persistent KV caching change TTFT, inter-token latency, output throughput and GPU memory as **input prompt length** increases? This follows the clarified input-length sweep, not a sweep over generated output lengths. All runs generate a fixed 128 tokens. Do not describe the resulting x-axis as generated sequence length.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-4B`, immutable HF revision recorded by downloader |
| Hardware | One Colab A100; actual GPU model, VRAM and software recorded |
| Precision | BF16 |
| Input prompt tokens | 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 |
| Output tokens | 128, including the token produced by prefill |
| Batch / concurrent requests | 1 |
| Sampling | Greedy, temperature 0, ignore EOS until 128 tokens |
| Random input | CPU `torch.Generator().manual_seed(42)`; uniformly sampled non-special token IDs |
| Global seeds | `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)` |
| Iterative scheduling | Enabled in both modes; one newly emitted token per iteration |
| CUDA graphs | Enabled for decode in both modes; prefill eager in both |
| Prefix reuse across requests | Disabled in both; warmup cannot satisfy a measured prompt from saved blocks |
| Chunked prefill | Not exercised: token budget covers the complete prompt/context in each case |
| Repetitions | 2 unmeasured full-request warmups, then 3 measured requests per case |
| Isolation | Fresh process for each (input length, cache mode); modes alternate order across lengths |
| KV pool policy | Explicit `ceil((prompt + output)/256)` blocks; does not reserve 90% of VRAM merely because it is available |

The 32,768 + 128 maximum requested length fits Qwen3-4B's 40,960-token native context. No YaRN or RoPE scaling is enabled. Every case saves the exact input token IDs and SHA-256 hash; both modes must have the same hash. Random token prompts control shapes but do not represent natural-language quality or typical serving traffic. Seeding fixes inputs; BF16 kernels are not promised bitwise deterministic across environments.

## What changed in the engine

- The existing Qwen model path now handles Qwen3's per-head Q/K RMSNorm, bias-free QKV projections, explicit `head_dim`, and tied or untied output embeddings. Qwen2 remains supported. Checkpoints load from one safetensors file or any shard count declared in the index.
- Cache allocation uses Qwen3's actual `head_dim=128`; `hidden_size/num_heads=80` would be incorrect for this checkpoint.
- `use_kv_cache=False` bypasses KV pool allocation and the block manager. Each iteration sends **all prompt tokens plus all previously generated tokens** through every layer. K/V are recomputed, used within the current attention operation and never reused for the next iteration.
- No-cache execution retains the existing model, fused projections, normalization/MLP path, FlashAttention, sampler, iterative request loop and optional CUDA graphs. Prefix reuse and chunked prefill need persistent K/V and therefore cannot apply to this mode.
- Existing `EngineCore(...).generate(...)` usage remains supported. `from nanovllm import LLM, SamplingParams` is an alias for convenience, not a promise of complete upstream nano-vLLM/vLLM API compatibility.
- The benchmark's host timer stops only after sampled token IDs are available on CPU (`Tensor.tolist()` synchronizes the relevant GPU work). Tokenization, model loading, compilation, graph capture and warmups are outside measurement.

### CUDA graph caveat: padding is real work

The no-cache graph has one fixed capacity per case: **prompt length + output length**. It recomputes that many positions on every decode replay, samples at `current_context_length - 1`, and uses causal attention so future padded positions cannot change the sampled logits. It retains input/activation buffers required by CUDA graphs, but does not reuse computed K/V across iterations.

This avoids capturing a new graph inside measured decoding, and both modes really replay CUDA graphs. It also adds padding work, especially at small prompt lengths. For a 64-token prompt and 128-token output, the no-cache graph computes 192 positions per decode while the real context grows from 65 to 191. Treat those results as **this graph implementation's cache ablation**, not a pure mathematical estimate of cache savings. `graph_capacity`, `graph_replays` and `processed_token_positions` make this visible. The eager control switches graphs off in **both** modes and uses exact context lengths; run it as a sensitivity check.

No-cache graphs currently support `max_num_seqs=1`. Eager no-cache supports packed multiple requests subject to the full-prefix token budget. The runner raises on unsupported graph batching and over-budget contexts instead of silently dropping an optimization or hanging.

## Colab A100 reproduction

Select **Runtime → Change runtime type → A100 GPU**. Use the supplied notebook for executable cells, or run the following from a shell in your GPU environment. The version choices below are a proposed reproducible environment; CUDA execution still needs the included validation gate. FlashAttention may build from source and take several minutes.

```bash
git clone --branch feat/qwen3-kv-cache-ablation https://github.com/SnorlaxPapa/tiny-vllm.git
cd tiny-vllm
nvidia-smi
python -m pip install uv
uv venv --python 3.11 .venv-benchmark
uv pip install --python .venv-benchmark/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-benchmark/bin/python -r requirements-benchmark.txt setuptools wheel psutil
MAX_JOBS=2 uv pip install --python .venv-benchmark/bin/python flash-attn==2.7.4.post1 --no-build-isolation
```

The isolated environment avoids conflicts with Colab's preinstalled torchvision/vLLM stack. A compatible NVIDIA driver and CUDA toolkit are required if FlashAttention builds from source. Do not replace one arm with a different attention backend to get past an installation error. All actual installed versions are written to each environment JSON.

Download the checkpoint and resolve its revision once:

```bash
.venv-benchmark/bin/python benchmarks/kv_cache/prepare_model.py
```

This creates `checkpoints/Qwen3-4B` and `a100.local.json`, `smoke.local.json`, `eager.local.json` beside the scripts. To reproduce a previous run exactly, pass its recorded SHA using `--revision <sha>`. Keep that checkpoint directory dedicated to the selected model revision.

Run tests, then GPU correctness validation **before trusting any performance results**:

```bash
.venv-benchmark/bin/python -m pytest -q tests
.venv-benchmark/bin/python benchmarks/kv_cache/validate_gpu.py \
  --model checkpoints/Qwen3-4B --out results/validation.json
```

The GPU validator compares logits against Hugging Face with identical teacher-forced histories, then compares cache/no-cache and eager/graph modes. It exercises prompt lengths 7, 31, 255, 256 and 257, including growth across a 256-token KV block boundary and different sample positions within the padded graph. It asserts zero allocated KV tensors in no-cache mode, correct graph replay counts, and public API token counts across repeated requests. The JSON records numerical errors and top-1 agreement. BF16 comparisons use `atol=0.3`, `rtol=0.02`; an assertion failure must be investigated, not hidden by widening tolerances. The validator does not prove numerical equivalence at every 32K position; add longer validation cases if required.

Start with the short smoke run (64/256 prompts, 8 output tokens, one warmup/repeat):

```bash
.venv-benchmark/bin/python benchmarks/kv_cache/run.py \
  --config benchmarks/kv_cache/smoke.local.json --out results/smoke
```

Then run the full study:

```bash
.venv-benchmark/bin/python benchmarks/kv_cache/run.py \
  --config benchmarks/kv_cache/a100.local.json --out results/kv_cache_a100
```

The no-cache 16K/32K cases can be expensive. The harness preserves completed repeats and records OOM/other failures; it never invents a row or changes output length to make a case succeed. Keep the GPU idle, avoid simultaneous workloads, and monitor Colab runtime limits. Re-running with the same nonempty output directory is rejected to protect measurements; use a new directory. If a session is interrupted, plot the saved partial data with the command below. For targeted retries, copy the configuration and restrict `prompt_lengths`; keep both modes and write to a new output directory.

Run the eager control, optionally restricting its prompt lengths in a copied config:

```bash
.venv-benchmark/bin/python benchmarks/kv_cache/run.py \
  --config benchmarks/kv_cache/eager.local.json --out results/kv_cache_eager
```

Rebuild plots/report from saved measurements at any time:

```bash
.venv-benchmark/bin/python benchmarks/kv_cache/plot.py --results results/kv_cache_a100
```

## Public API example

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "checkpoints/Qwen3-4B", benchmark=True,
    use_kv_cache=False, enable_prefix_caching=False,
    enforce_eager=False, max_num_seqs=1,
    max_model_len=192, max_num_batch_tokens=192,
    no_cache_graph_capacity=192,
)
params = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
ttft_s, mean_itl_s = llm.generate([[100] * 64], params)
print(llm.last_metrics)
print(llm.output_token_ids)
```

For the cached arm set `use_kv_cache=True`; the benchmark additionally sizes the KV pool explicitly. Keep the other settings equal. For regular text generation, omit `benchmark=True` to return decoded strings. Tokenized inputs bypass chat templates and preserve the exact requested prompt length.

## Metric definitions and interpretation

Let `t0` be the generation loop start, `t1` the first CPU-visible output token, `tN` the last, and `N=128`.

| CSV column | Definition | Interpretation |
|---|---|---|
| `ttft_ms` | `1000 × (t1 − t0)` | First-token latency; mostly prompt processing in this batch-1 experiment |
| `mean_itl_ms` | Mean of `1000 × (ti − t(i−1))`, `i=2..N` | Subsequent-token latency; excludes TTFT |
| `p50/p95/p99_itl_ms` | Linearly interpolated per-request quantiles of individual gaps | Token latency variability; not a percentile of run averages |
| `output_tokens_per_s` | `N / (tN − t0)` | End-to-end output throughput; includes prefill time |
| `decode_tokens_per_s` | `(N−1) / (tN − t1)` | Decode-only throughput; at batch 1, inverse mean ITL |
| `peak_allocated_bytes` | PyTorch peak live tensor allocation during measured request | Includes model, KV pool if any, graph-retained tensors and activations |
| `peak_reserved_bytes` | PyTorch peak allocator reservation during measured request | Includes reusable allocator/graph pools retained after warmup |
| `allocated/reserved_bytes` | End-of-request PyTorch allocation/reservation | Excludes some driver/runtime allocations; not total VRAM usage |
| `kv_pool_bytes` | Actual sizes of all persistent K and V tensors | Allocated cache capacity; zero for no-cache |
| `logical_kv_bytes_at_last_step` | `2 × layers × KV heads × head_dim × 2 bytes × (P+N−1)` for cached mode | Analytic logical K/V size at final forward; not measured traffic; zero for no-cache |
| `processed_token_positions` | Cached: `P+N−1`; eager no-cache: `NP+N(N−1)/2`; graph no-cache: `P+(N−1)C` | Algorithmic work proxy through layers; **not FLOPs**; `C` is padded graph capacity |

**ITL is not memory consumption or effective bandwidth.** Decode with KV caching can be bandwidth-bound, but uncached full-prefix recomputation can become compute-bound. Do not report GB/s derived from ITL alone: it requires measured or explicitly modeled bytes transferred. Throughput likewise is not a direct measure of GPU compute utilization. For hardware bandwidth/FLOPs claims, profile selected cases separately with NVIDIA tooling and keep profiling overhead outside the benchmark run.

For Qwen3-4B BF16, logical K/V storage is `2 × 36 × 8 × 128 × 2 = 147,456` bytes per context token (144 KiB). Pool allocation is rounded up to 256-token blocks. CUDA graph private pools can keep no-cache activation memory reserved; no persistent KV cache does **not** imply no persistent GPU memory or necessarily a lower measured peak.

TTFT should not be expected to show the same benefit as decode: with prefix reuse disabled, both modes process the full initial prompt. Cached mode also writes K/V into its pool. These are hypotheses to examine, not measured conclusions.

## Generated deliverables

Each result directory contains:

- `raw.csv`: one measured row per repeat, plus explicit error rows if any.
- `summary.csv`: per-mode/per-input medians, minima and maxima across successful repeats.
- `metrics.png` and `metrics.svg`: six panels covering latency, throughput and memory; min–max error bars, not confidence intervals.
- `README.md`: measured metric table, plots, configuration summary, coverage gaps and interpretation limits.
- `config.json`: exact experiment settings including resolved model revision.
- `prompts/<length>.json`: exact random input IDs used by both modes.
- `runs/*_tokens.csv`: every output token's index, context length, elapsed time and ITL.
- `runs/*.jsonl`: incrementally flushed per-repeat records, retained even if a later repeat fails.
- `runs/*_output_<repeat>.json`: actual generated token IDs.
- `runs/*_environment.json`: GPU properties, driver output, CUDA/Python/package versions, checkpoint config and Git commit.
- `logs/*.log`: full worker logs, including tracebacks for failed cases.

The report only plots successful measurements and explicitly reports incomplete coverage. Greedy output may diverge on near-tied BF16 logits between kernels, despite identical prompts; saved output IDs and teacher-forced validation make that inspectable. Timing summaries alone do not establish causal hardware utilization or represent all serving engines.

To submit the final project, review `results/kv_cache_a100/README.md`, include the validation JSON, raw/summary CSVs and figures, and attach or commit the result directory. Generated results and checkpoints are ignored by Git; use `git add -f` for the specific results you intend to publish, **never** for model weights. The notebook downloads a ZIP containing the run artifacts.

## Validation performed before GPU execution

CPU tests cover full-prefix iterative scheduling, EOS, cache block growth/deallocation, prevention of cross-request prefix reuse, explicit failure on impossible budgets, metric formulas and failure-aware aggregation. Small randomly initialized Qwen3 models are compared against Hugging Face in FP32 using PyTorch SDPA as a CPU substitute for FlashAttention; tests cover nonstandard head dimensions, Q/K normalization, single/sharded checkpoints, tied/untied embeddings and causal padding. This does **not** validate CUDA kernels, capture, GPU memory usage or performance. Use the A100 validation gate for those execution paths.

## Primary references

- [Qwen3-4B model configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json).
- [Hugging Face Qwen3 reference implementation, Transformers 4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py).
- [FlashAttention source and installation documentation](https://github.com/Dao-AILab/flash-attention).
- [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).
