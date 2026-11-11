"""A100 correctness gate: HF reference, KV/eager, full-prefix/eager and both graphs.

Teacher-forced token histories ensure every implementation sees identical input.
This is a correctness test, not a performance benchmark.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from transformers import AutoModelForCausalLM
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.sequence import Sequence
    from nanovllm.layers.rope import get_rope

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, default=Path("validation.json"))
    parser.add_argument("--atol", type=float, default=0.3)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for validation")
    torch.manual_seed(42)
    # Cross both padded graph positions and the 256-token cache block boundary.
    lengths, steps, capacity = [7, 31, 255, 256, 257], 4, 272
    reference = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    histories = {n: torch.randint(100, 10000, (n+steps,), generator=torch.Generator().manual_seed(42)).tolist() for n in lengths}
    expected = {}
    with torch.inference_mode():
        for n, ids in histories.items():
            for step in range(steps):
                logits = reference(torch.tensor([ids[:n+step]], device="cuda"), use_cache=False).logits[0, -1]
                expected[n, step] = logits.float().cpu()
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    report = []
    baseline = {}
    for cache, graphs in [(False, False), (True, False), (False, True), (True, True)]:
        engine = LLM(args.model, benchmark=True, use_kv_cache=cache,
                     enable_prefix_caching=False, enforce_eager=not graphs,
                     max_num_seqs=1, max_model_len=capacity,
                     max_num_batch_tokens=capacity, no_cache_graph_capacity=capacity,
                     num_kvcache_blocks=math.ceil(capacity/256) if cache else 0)
        if not cache:
            assert engine.scheduler.block_manager is None
            assert all(layer.self_attn.attn.k_cache.numel() == 0 and layer.self_attn.attn.v_cache.numel() == 0 for layer in engine.runner.model.layers)
        for n, ids in histories.items():
            seq = Sequence(ids[:n], SamplingParams(temperature=0, max_tokens=steps, ignore_eos=True))
            engine.scheduler.add(seq)
            start_replays = engine.runner.graph_replays
            for step in range(steps):
                scheduled, prefill = engine.scheduler.schedule()
                actual = engine.runner.run(scheduled, prefill, return_logits=True)[0].detach().float().cpu()
                torch.testing.assert_close(actual, expected[n, step], atol=args.atol, rtol=args.rtol)
                if not cache and not graphs:
                    baseline[n, step] = actual
                else:
                    torch.testing.assert_close(actual, baseline[n, step], atol=args.atol, rtol=args.rtol)
                report.append(dict(cache=cache, graphs=graphs, prompt_tokens=n, step=step,
                                   max_abs_error_vs_hf=(actual-expected[n, step]).abs().max().item(),
                                   top1_matches_hf=actual.argmax().item() == expected[n, step].argmax().item()))
                # Use the same next token in every mode (not model-specific argmax).
                engine.scheduler.postprocess(scheduled, [ids[n+step]], prefill)
            assert engine.scheduler.is_finished()
            assert engine.runner.graph_replays-start_replays == (steps-1 if graphs else 0)
        # Exercise the public API too, including metrics/replay counts and reuse.
        for _ in range(2):
            engine.generate([histories[31][:31]], SamplingParams(temperature=0, max_tokens=steps, ignore_eos=True))
            assert engine.last_metrics["output_tokens"] == steps
            assert len(engine.last_metrics["token_events"]) == steps
            assert len(engine.itls) == steps-1
            assert engine.last_metrics["graph_replays"] == (steps-1 if graphs else 0)
        del engine
        get_rope.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(status="passed", atol=args.atol, rtol=args.rtol,
                                      checks=report), indent=2)+"\n")
    print(f"Passed {len(report)} teacher-forced comparisons; wrote {args.out}")


if __name__ == "__main__":
    main()
