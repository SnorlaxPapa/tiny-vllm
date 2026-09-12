import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from itertools import count
from statistics import median
from time import perf_counter

import pandas as pd
import torch
import vllm

from vllm import EngineArgs, LLMEngine
from vllm import SamplingParams as VLLMSamplingParams
from vllm.sampling_params import RequestOutputKind


MODEL_DIR = "checkpoints"
CSV_PATH = "vllm_results.csv"

MAX_MODEL_LEN = 4096
OUTPUT_LENGTH = 64
REPEATS = 10

rng = torch.Generator(device="cpu").manual_seed(42)
request_counter = count()

sampling = VLLMSamplingParams(
    temperature=0.0,
    max_tokens=OUTPUT_LENGTH,
    ignore_eos=True,
    detokenize=False,
    output_kind=RequestOutputKind.CUMULATIVE,
)


def create_batch(batch_size: int, seq_len: int) -> list[list[int]]:
    return torch.randint(
        low=0,
        high=151600,
        size=(batch_size, seq_len),
        generator=rng,
        device="cpu",
    ).tolist()


def run_batch(engine, prompts) -> tuple[float, float, float]:
    if not prompts:
        raise ValueError("Empty batch.")

    if engine.has_unfinished_requests():
        raise RuntimeError("Previous requests have not finished.")

    token_counts = {}
    first_token_times = {}
    last_token_times = {}
    itls = []

    torch.cuda.synchronize()
    total_start = perf_counter()

    for prompt in prompts:
        request_id = str(next(request_counter))
        token_counts[request_id] = 0

        engine.add_request(
            request_id,
            {"prompt_token_ids": prompt},
            sampling,
        )

    batch_start = perf_counter()

    while engine.has_unfinished_requests():
        outputs = engine.step()
        timestamp = perf_counter()

        for output in outputs:
            if not output.outputs:
                continue

            request_id = output.request_id
            generated = len(output.outputs[0].token_ids)
            previous = token_counts[request_id]
            new_tokens = generated - previous

            if new_tokens == 0:
                continue

            if new_tokens != 1:
                raise RuntimeError(
                    f"Received {new_tokens} tokens in one update; "
                    "cannot measure individual ITL."
                )

            if previous == 0:
                first_token_times[request_id] = timestamp
            else:
                itls.append(
                    timestamp - last_token_times[request_id]
                )

            last_token_times[request_id] = timestamp
            token_counts[request_id] = generated

    torch.cuda.synchronize()
    total_elapsed = perf_counter() - total_start

    if any(n != OUTPUT_LENGTH for n in token_counts.values()):
        raise RuntimeError(
            f"Unexpected output lengths: {list(token_counts.values())}"
        )

    if len(first_token_times) != len(prompts):
        raise RuntimeError("Missing first-token measurements.")

    expected_intervals = len(prompts) * (OUTPUT_LENGTH - 1)
    if len(itls) != expected_intervals:
        raise RuntimeError("Missing inter-token measurements.")

    average_ttft = sum(
        timestamp - batch_start
        for timestamp in first_token_times.values()
    ) / len(prompts)

    average_itl = sum(itls) / len(itls)

    return average_ttft, average_itl, total_elapsed


def measure_case(
    engine,
    batch_size: int,
    seq_len: int,
) -> tuple[float, float, float]:
    if seq_len + OUTPUT_LENGTH > MAX_MODEL_LEN:
        raise ValueError("Input plus output exceeds max_model_len.")

    run_batch(engine, create_batch(batch_size, seq_len))

    ttfts = []
    itls = []
    elapsed_times = []

    for _ in range(REPEATS):
        prompts = create_batch(batch_size, seq_len)

        ttft, itl, elapsed = run_batch(engine, prompts)

        ttfts.append(ttft)
        itls.append(itl)
        elapsed_times.append(elapsed)

    output_throughput = (
        batch_size * OUTPUT_LENGTH
    ) / median(elapsed_times)

    return median(ttfts), median(itls), output_throughput


def benchmark_vllm(
    engine,
    csv_path: str,
) -> pd.DataFrame:
    rng.manual_seed(42)

    batches = [1, 2, 4, 8, 16, 32, 64]
    sequences = [
        32, 64, 128, 256, 512, 1024,
        1500, 2048, 3000, 3600, 3800, 4030,
    ]

    cases = [
        ("sequence_length", 4, seq_len)
        for seq_len in sequences
    ] + [
        ("batch_size", batch_size, 512)
        for batch_size in batches
    ]

    os.makedirs(
        os.path.dirname(os.path.abspath(csv_path)),
        exist_ok=True,
    )

    rows = []

    for sweep, batch_size, seq_len in cases:
        print(
            f"Benchmarking batch={batch_size}, "
            f"sequence={seq_len}",
            flush=True,
        )

        ttft, itl, throughput = measure_case(
            engine,
            batch_size,
            seq_len,
        )

        rows.append({
            "engine": "vLLM",
            "sweep": sweep,
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "output_length": OUTPUT_LENGTH,
            "ttft_ms": ttft * 1000,
            "itl_ms": itl * 1000,
            "output_throughput_tps": throughput,
        })

        results = pd.DataFrame(rows)
        results.to_csv(csv_path, index=False)

        print(
            f"TTFT={ttft * 1000:.2f} ms, "
            f"ITL={itl * 1000:.2f} ms, "
            f"throughput={throughput:.2f} tokens/s",
            flush=True,
        )

    print(f"Saved to {csv_path}", flush=True)
    return results


def main():
    print(f"vLLM version: {vllm.__version__}", flush=True)
    print(
        f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}",
        flush=True,
    )

    engine_args = EngineArgs(
        model=MODEL_DIR,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=512,
        max_num_batched_tokens=16384,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=False,
        skip_tokenizer_init=True,
        disable_log_stats=True,
        seed=42,
    )

    engine = LLMEngine.from_engine_args(engine_args)
    benchmark_vllm(engine, CSV_PATH)


if __name__ == "__main__":
    main()