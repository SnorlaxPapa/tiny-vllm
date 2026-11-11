"""CPU tests for scheduling, configuration, and metric accounting."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from benchmarks.kv_cache.run import percentile, processed_positions, validate_config
from benchmarks.kv_cache.plot import summarize


def config(cache=False, **kwargs):
    values = dict(use_kv_cache=cache, enable_prefix_caching=False, max_num_batch_tokens=1024,
                  max_num_seqs=2, eos=99, num_kvcache_blocks=8, kv_cache_block_size=256)
    return SimpleNamespace(**(values | kwargs))


@pytest.mark.parametrize("cache", [False, True])
def test_iterative_scheduler_and_block_boundary(cache):
    scheduler = Scheduler(config(cache))
    sequences = [Sequence(list(range(255)), SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)),
                 Sequence([1, 2], SamplingParams(max_tokens=2, ignore_eos=True))]
    for seq in sequences:
        scheduler.add(seq)
    calls = 0
    while not scheduler.is_finished():
        batch, prefill = scheduler.schedule()
        assert batch
        if not cache:
            assert scheduler.block_manager is None
            assert all(seq.num_computed_tokens == 0 and seq.num_scheduled_tokens == len(seq) and not seq.block_list for seq in batch)
        scheduler.postprocess(batch, [3]*len(batch), prefill)
        calls += 1
        assert calls < 10
    assert [seq.num_completion_tokens for seq in sequences] == [4, 2]
    if cache:
        assert len(scheduler.block_manager.free_block_ids) == 8
        assert not scheduler.block_manager.hash_to_block_id


def test_no_cache_eos_and_ignore_eos():
    scheduler = Scheduler(config())
    a = Sequence([1], SamplingParams(max_tokens=5, ignore_eos=False))
    b = Sequence([2], SamplingParams(max_tokens=2, ignore_eos=True))
    scheduler.add(a)
    scheduler.add(b)
    batch, prefill = scheduler.schedule()
    scheduler.postprocess(batch, [99, 99], prefill)
    assert a.num_completion_tokens == 1
    batch, prefill = scheduler.schedule()
    assert batch == [b]
    scheduler.postprocess(batch, [99], prefill)
    assert scheduler.is_finished()


def test_no_cache_budget_fails_instead_of_hanging():
    scheduler = Scheduler(config(max_num_batch_tokens=2))
    scheduler.add(Sequence([1, 2, 3]))
    with pytest.raises(RuntimeError, match="budget"):
        scheduler.schedule()


def test_insufficient_cache_fails_instead_of_hanging():
    scheduler = Scheduler(config(True, num_kvcache_blocks=1))
    scheduler.add(Sequence([1]*257))
    with pytest.raises(RuntimeError, match="progress"):
        scheduler.schedule()


def test_no_prefix_reuse_between_requests():
    scheduler = Scheduler(config(True))
    for _ in range(2):
        seq = Sequence([1]*300, SamplingParams(max_tokens=1))
        scheduler.add(seq)
        batch, prefill = scheduler.schedule()
        assert seq.num_computed_tokens == 0
        assert seq.num_scheduled_tokens == 300
        scheduler.postprocess(batch, [2], prefill)
        assert scheduler.is_finished()


def test_metric_arithmetic():
    assert percentile([1, 2, 3, 4], .5) == 2.5
    assert percentile([], .5) is None
    assert processed_positions(64, 128, True) == 191
    assert processed_positions(64, 128, False) == sum(range(64, 192))
    assert processed_positions(64, 128, False, 192) == 64+127*192


def test_configuration_rejects_invalid_sweep():
    c = json.loads(Path("benchmarks/kv_cache/a100.json").read_text())
    validate_config(c)
    for key, bad in [("output_tokens", 1), ("repeats", 0), ("warmups", 0), ("prompt_lengths", [64, 64])]:
        with pytest.raises(ValueError):
            validate_config(c | {key: bad})


def test_summary_does_not_turn_failures_into_measurements():
    from benchmarks.kv_cache.plot import METRICS
    rows = [dict(status="error", mode="no_kv", prompt_tokens="64")]
    for value in (1, 3, 8):
        rows.append(dict(status="ok", mode="kv", prompt_tokens="64", **{key: str(value) for key in METRICS}))
    result = summarize(rows)
    assert len(result) == 1
    assert result[0]["repeats"] == 3
    assert result[0]["ttft_ms"] == 3
    assert result[0]["ttft_ms_min"] == 1
    assert result[0]["ttft_ms_max"] == 8


def test_preemption_can_resume_after_freeing_blocks():
    scheduler = Scheduler(config(True, num_kvcache_blocks=2))
    a = Sequence([1]*256, SamplingParams(max_tokens=2, ignore_eos=True))
    b = Sequence([2]*256, SamplingParams(max_tokens=2, ignore_eos=True))
    scheduler.add(a)
    scheduler.add(b)
    steps = 0
    while not scheduler.is_finished():
        batch, prefill = scheduler.schedule()
        scheduler.postprocess(batch, [3]*len(batch), prefill)
        steps += 1
        assert steps < 12
    assert a.num_completion_tokens == b.num_completion_tokens == 2


def test_plot_and_report_handle_partial_results(tmp_path):
    from benchmarks.kv_cache.plot import METRICS, report
    from benchmarks.kv_cache.run import FIELDS
    config = dict(output_tokens=8, cuda_graphs=True, repeats=1, prompt_lengths=[64, 128])
    (tmp_path / "config.json").write_text(json.dumps(config))
    with (tmp_path / "raw.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for mode in ("kv", "no_kv"):
            writer.writerow(dict(status="ok", mode=mode, prompt_tokens=64,
                                 **{key: 1. for key in METRICS}))
        writer.writerow(dict(status="error", mode="no_kv", prompt_tokens=128, error="synthetic test failure"))
    report(tmp_path)
    assert (tmp_path / "metrics.png").stat().st_size > 1000
    assert (tmp_path / "metrics.svg").stat().st_size > 1000
    body = (tmp_path / "README.md").read_text()
    assert "Incomplete: prompt=128" in body
    assert "synthetic test failure" in body

    # A killed worker may have flushed JSONL before the parent writes raw.csv.
    saved_rows = list(csv.DictReader((tmp_path / "raw.csv").open()))
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "partial.jsonl").write_text("".join(json.dumps(r)+"\n" for r in saved_rows))
    (tmp_path / "raw.csv").unlink()
    report(tmp_path)
    assert (tmp_path / "raw.csv").exists()
    assert "Incomplete: prompt=128" in (tmp_path / "README.md").read_text()
