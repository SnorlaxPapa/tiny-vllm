
from statistics import median
import pandas as pd

from nanovllm.engine.engine import EngineCore
from nanovllm.sampling_params import SamplingParams
import torch
import matplotlib.pyplot as plt
import torch._dynamo

torch._dynamo.config.recompile_limit = 64

torch.manual_seed(42)
model_dir = "/content/drive/MyDrive/vllmproject/checkpoints"
engine = EngineCore(model_dir, benchmark=True)
rng = torch.Generator(device="cpu").manual_seed(42)

def create_batch(batch_size: int, seq_len: int) -> list[list[int]]:
    return torch.randint(
        low=0,
        high=151600,
        size=(batch_size, seq_len),
        generator=rng,
        device="cpu",
    ).tolist()

def measure_case(
    engine: EngineCore,
    batch_size: int,
    seq_len: int,
    sampling: SamplingParams,
    repeats: int = 10,
) -> tuple[float, float]:

    # Warm up this exact batch/sequence shape.
    warmup_batch = create_batch(batch_size, seq_len)
    engine.generate(warmup_batch, sampling)

    ttft_results = []
    itl_results = []

    # Measure using new random prompts.
    for _ in range(repeats):
        batch = create_batch(batch_size, seq_len)
        ttft, itl = engine.generate(batch, sampling)

        ttft_results.append(ttft)
        itl_results.append(itl)

    return median(ttft_results), median(itl_results)


def benchmark_batch(
    engine: EngineCore,
    batches: list[int],
    seq_len: int,
) -> tuple[list[float], list[float]]:

    ttfts = []
    itls = []

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        ignore_eos=True,
    )

    for batch_size in batches:
        ttft, itl = measure_case(
            engine=engine,
            batch_size=batch_size,
            seq_len=seq_len,
            sampling=sampling,
        )

        ttfts.append(ttft)
        itls.append(itl)

        print(
            f"batch={batch_size}, "
            f"TTFT={ttft * 1000:.2f} ms, "
            f"ITL={itl * 1000:.2f} ms"
        )

    return ttfts, itls


def benchmark_seq(
    engine: EngineCore,
    batch_size: int,
    seq_lengths: list[int],
) -> tuple[list[float], list[float]]:

    ttfts = []
    itls = []

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        ignore_eos=True,
    )

    for seq_len in seq_lengths:
        ttft, itl = measure_case(
            engine=engine,
            batch_size=batch_size,
            seq_len=seq_len,
            sampling=sampling,
        )

        ttfts.append(ttft)
        itls.append(itl)

        print(
            f"sequence={seq_len}, "
            f"TTFT={ttft * 1000:.2f} ms, "
            f"ITL={itl * 1000:.2f} ms"
        )

    return ttfts, itls


def benchmark_nanovllm(engine, csv_path: str="results.csv"):
    rng.manual_seed(42)
    batches = [1, 2, 4, 8, 16, 32, 64]
    fixed_seq_len = 512

    fixed_batch_size = 4
    sequences = [
        32, 64, 128, 256, 512, 1024,
        1500, 2048, 3000, 3600, 3800, 4030,
    ]

    print("Benchmarking across sequences for NanoVLLM!")

    # These are still measured in seconds.
    ttft_seq, itl_seq = benchmark_seq(
        engine,
        fixed_batch_size,
        sequences,
    )

    print("Benchmarking across batches for NanoVLLM!")

    ttft_bs, itl_bs = benchmark_batch(
        engine,
        batches,
        fixed_seq_len,
    )

    rows = []

    # Sequence-length sweep
    for seq_len, ttft, itl in zip(
        sequences,
        ttft_seq,
        itl_seq,
    ):
        rows.append({
            "engine": "NanoVLLM",
            "sweep": "sequence_length",
            "batch_size": fixed_batch_size,
            "sequence_length": seq_len,
            "output_length": 64,
            "ttft_ms": ttft * 1000,
            "itl_ms": itl * 1000,
        })

    # Batch-size sweep
    for batch_size, ttft, itl in zip(
        batches,
        ttft_bs,
        itl_bs,
    ):
        rows.append({
            "engine": "NanoVLLM",
            "sweep": "batch_size",
            "batch_size": batch_size,
            "sequence_length": fixed_seq_len,
            "output_length": 64,
            "ttft_ms": ttft * 1000,
            "itl_ms": itl * 1000,
        })

    results = pd.DataFrame(rows)

    results.to_csv(csv_path, index=False)

    print(f"Saved results to {csv_path}")
    return results


nanovllm_results = benchmark_nanovllm(engine, "/content/drive/MyDrive/vllmproject/nanovllm_results.csv")
