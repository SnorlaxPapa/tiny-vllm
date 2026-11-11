"""Create summary.csv, plots, and a Markdown report from measured CSV rows only."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

METRICS = {
    "ttft_ms": "TTFT (ms)",
    "mean_itl_ms": "Mean inter-token latency (ms)",
    "output_tokens_per_s": "End-to-end output throughput (tokens/s)",
    "decode_tokens_per_s": "Decode-only throughput (tokens/s)",
    "peak_allocated_bytes": "Peak PyTorch allocation (GiB)",
    "peak_reserved_bytes": "Peak PyTorch reservation (GiB)",
}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            groups[(row["mode"], int(row["prompt_tokens"]))].append(row)
    summary = []
    for (mode, length), members in sorted(groups.items()):
        entry = dict(mode=mode, prompt_tokens=length, repeats=len(members))
        for metric in METRICS:
            values = [float(row[metric]) for row in members]
            entry[metric] = statistics.median(values)
            entry[metric+"_min"] = min(values)
            entry[metric+"_max"] = max(values)
        summary.append(entry)
    return summary


def report(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Recover repeats already flushed by a worker even if Colab interrupted
    # the parent before it could aggregate raw.csv.
    if (results / "runs").exists():
        from benchmarks.kv_cache.run import save_csv
        save_csv(results)
    rows = list(csv.DictReader((results / "raw.csv").open()))
    summary = summarize(rows)
    config = json.loads((results / "config.json").read_text())
    failures = [r for r in rows if r["status"] != "ok"]
    report_lines = ["# Qwen3-4B KV cache ablation: measured results", "",
        "Input prompt length varies; output length stays fixed. Batch size 1, BF16, greedy decoding, seed 42.",
        f"Output tokens: **{config['output_tokens']}**; CUDA graphs: **{config['cuda_graphs']}**; requested repeats: **{config['repeats']}**.", "",
        "TTFT and ITL are host-observed token latency. Throughput is not FLOPs. ITL is not measured bandwidth.",
        "Memory is PyTorch allocated/reserved memory, not total device usage. See per-case environment JSON and logs.", "",
        "No-cache graphs recompute at a fixed padded capacity of prompt + output tokens; eager runs use exact current lengths.",
        "Prefix reuse is disabled. Initialization, graph capture and warmup are excluded. Error bars show min–max across repeats, not confidence intervals.", "",
        "[Raw measured rows](raw.csv) · [Configuration](config.json) · [Per-case data and token timings](runs/)", ""]
    if not summary:
        report_lines += ["**No successful measurements.** Inspect logs; no performance claims or plots are available."]
    else:
        with (results / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
        for ax, (metric, label) in zip(axes.flat, METRICS.items()):
            scale = 2**30 if metric.endswith("bytes") else 1
            for mode, legend, color in [("kv", "KV cache", "#2563eb"), ("no_kv", "No KV cache", "#e56b25")]:
                members = sorted((r for r in summary if r["mode"] == mode), key=lambda r: r["prompt_tokens"])
                if not members:
                    continue
                x = [r["prompt_tokens"] for r in members]
                y = [r[metric]/scale for r in members]
                low = [(r[metric]-r[metric+"_min"])/scale for r in members]
                high = [(r[metric+"_max"]-r[metric])/scale for r in members]
                ax.errorbar(x, y, yerr=[low, high], marker="o", label=legend, color=color, capsize=3)
            ax.set_xscale("log", base=2)
            if not metric.endswith("bytes"):
                ax.set_yscale("log")
            ax.set_xlabel("Input prompt tokens")
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle("Qwen3-4B: KV cache ablation · median and min–max across repeats")
        fig.savefig(results / "metrics.png", dpi=180)
        fig.savefig(results / "metrics.svg")
        plt.close(fig)
        report_lines += ["![Measured metrics](metrics.png)", "", "[Summary CSV](summary.csv)", "",
            "| Prompt | Mode | Runs | TTFT ms | Mean ITL ms | Output tok/s | Decode tok/s | Peak allocated GiB |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|"]
        for r in sorted(summary, key=lambda r: (r["prompt_tokens"], r["mode"])):
            report_lines.append(f"| {r['prompt_tokens']} | {r['mode']} | {r['repeats']} | {r['ttft_ms']:.3f} | {r['mean_itl_ms']:.3f} | {r['output_tokens_per_s']:.2f} | {r['decode_tokens_per_s']:.2f} | {r['peak_allocated_bytes']/2**30:.3f} |")
    if failures:
        report_lines += ["", "## Failed cases", "", "Failures are missing data, never zero-valued measurements.", ""]
        report_lines += [f"- {r['prompt_tokens']} tokens, {r['mode']}: {r['error']}" for r in failures]
    report_lines += ["", "## Coverage", ""]
    for length in config["prompt_lengths"]:
        for mode in ("kv", "no_kv"):
            count = sum(r["status"] == "ok" and int(r["prompt_tokens"]) == length and r["mode"] == mode for r in rows)
            if count != config["repeats"]:
                report_lines.append(f"- Incomplete: prompt={length}, mode={mode}: {count}/{config['repeats']} successful repeats.")
    if not any(line.startswith("- Incomplete:") for line in report_lines):
        report_lines.append("All configured cases have the requested number of successful repeats.")
    report_lines += ["", "## Interpretation limits", "",
        "This sweeps **input length**, not generated output length. It measures how initial context length affects subsequent decoding over a fixed output window.",
        "No-cache attention recomputes the full prefix and can become compute-bound. Neither ITL nor its inverse establishes memory-bandwidth utilization.",
        "The small-prompt graph results include substantial padding overhead. Use the eager control to distinguish recomputation cost from padding/graph effects.",
        "GPU clocks, thermals and other workloads may affect measurements. This is one model, engine and GPU configuration, not a benchmark of the overall serving frontier."]
    (results / "README.md").write_text("\n".join(report_lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    report(parser.parse_args().results)
