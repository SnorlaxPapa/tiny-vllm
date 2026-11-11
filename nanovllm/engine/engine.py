from time import perf_counter

import torch
from transformers import AutoTokenizer

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class EngineCore:
    def __init__(self, model_dir: str, benchmark: bool = False, **config_kwargs):
        self.runner = ModelRunner(model_dir, **config_kwargs)
        config = self.runner.config
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.request_ids = []
        self.results = {}
        self.benchmark = benchmark
        self.last_metrics = {}
        self.output_token_ids = []

    def add_prompt(self, input: str | list[int], sampling_param: SamplingParams):
        if isinstance(input, str):
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
            )
        else:
            token_ids = input.copy()
        config = self.runner.config
        if not token_ids or sampling_param.max_tokens < 1:
            raise ValueError("Prompts and generated token counts must be nonempty")
        if len(token_ids) + sampling_param.max_tokens > config.max_model_len:
            raise ValueError("Prompt + requested output exceeds max_model_len")
        if any(not isinstance(t, int) or not 0 <= t < config.hf_config.vocab_size for t in token_ids):
            raise ValueError("Input token ID is outside the model vocabulary")
        sequence = Sequence(token_ids, sampling_param)
        self.request_ids.append(sequence.seq_id)
        self.scheduler.add(sequence)

    def generate(self, inputs, sampling_params):
        """Generate with the original API; benchmark=True still returns (TTFT, ITL).

        Rich metrics and raw token timestamps are available in last_metrics.
        Timings exclude tokenization/loading/warmup and stop at CPU-visible token
        IDs, after runner.run's blocking .tolist(). Units are seconds.
        """
        if not inputs:
            raise ValueError("At least one prompt is required")
        if not self.scheduler.is_finished():
            raise RuntimeError("Previous requests are still pending")
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(inputs):
                raise ValueError("Supply one sampling configuration per prompt")
        else:
            sampling_params = [sampling_params] * len(inputs)
        self.request_ids, self.results = [], {}
        self.output_token_ids = []
        # Validate each request before enqueueing any of them.
        try:
            for prompt, params in zip(inputs, sampling_params):
                self.add_prompt(prompt, params)
        except Exception:
            self.scheduler.waiting.clear()
            self.request_ids.clear()
            raise
        self.ttft, self.first_token_times, self.last_token_times = {}, {}, {}
        self.itls = []
        events, completed = [], {}
        replay_start = self.runner.graph_replays
        if self.benchmark:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        self.start_times = {sid: start for sid in self.request_ids}
        final_token_time = start
        while not self.scheduler.is_finished():
            sequences, is_prefill = self.scheduler.schedule()
            if not sequences:
                raise RuntimeError("Scheduler made no progress")
            outputs = self.runner.run(sequences, is_prefill)
            now = perf_counter()
            for seq in sequences:
                produces_token = not is_prefill or seq.num_computed_tokens + seq.num_scheduled_tokens == len(seq)
                if not produces_token:
                    continue
                itl = None
                if seq.seq_id not in self.first_token_times:
                    self.first_token_times[seq.seq_id] = now
                    self.ttft[seq.seq_id] = now - start
                else:
                    itl = now - self.last_token_times[seq.seq_id]
                    self.itls.append(itl)
                self.last_token_times[seq.seq_id] = now
                final_token_time = now
                if self.benchmark:
                    events.append(dict(request_id=seq.seq_id,
                                       output_index=seq.num_completion_tokens,
                                       context_tokens=len(seq),
                                       elapsed_s=now-start, itl_s=itl))
            self.scheduler.postprocess(sequences, outputs, is_prefill)
            for seq in sequences:
                if seq.status == SequenceStatus.FINISHED:
                    completed[seq.seq_id] = seq.token_ids[seq.num_prompt_tokens:]
        self.output_token_ids = [completed[sid] for sid in self.request_ids]
        elapsed = final_token_time - start
        total_tokens = sum(map(len, self.output_token_ids))
        mean_ttft = sum(self.ttft.values()) / len(self.ttft)
        mean_itl = sum(self.itls) / len(self.itls) if self.itls else None
        self.last_metrics = dict(
            ttft_s=mean_ttft, mean_itl_s=mean_itl, elapsed_s=elapsed,
            output_tokens=total_tokens, output_tokens_per_s=total_tokens/elapsed,
            graph_replays=self.runner.graph_replays-replay_start,
            token_events=events,
        )
        if self.benchmark:
            self.last_metrics.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
            )
            return mean_ttft, mean_itl
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in self.output_token_ids]
