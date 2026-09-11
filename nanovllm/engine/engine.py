from transformers import AutoTokenizer
from time import perf_counter
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.layers.sampler import Sampler

class EngineCore:

  def __init__(self, model_dir: str, benchmark: bool = False):
    self.runner = ModelRunner(model_dir) #allocate kv cache 
    config = self.runner.config
    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config.eos = self.tokenizer.eos_token_id #get eos marker
    self.scheduler = Scheduler(config)
    self.request_ids = []
    self.results = {}
    self.benchmark = benchmark
    
    #calculate tft for benchmark
    if self.benchmark:
      self.start_times = {}
      self.ttft = {}
      self.first_token_times = {}
      self.last_token_times = {}
      self.itls = []

  def add_prompt(self, input: str, sampling_param: SamplingParams):
    if self.benchmark: 
      start_time = perf_counter()

    token_ids = self.tokenizer.apply_chat_template(
      [{"role": "system", "content": "/no_think You are Qwen, a helpful assistant."},
       {"role": "user", "content": input}],
      tokenize=True,
      add_generation_prompt=True,
    )

    sequence = Sequence(token_ids["input_ids"], sampling_param)
    
    if self.benchmark:
      self.start_times[sequence.seq_id] = start_time

    self.request_ids.append(sequence.seq_id)
    self.scheduler.add(sequence)


  def generate(
      self,
      inputs: list[str],
      sampling_params: SamplingParams | list[SamplingParams],
  ):
    if isinstance(sampling_params, list):
      assert len(sampling_params) == len(inputs), "Ensure every prompt has an allocated sampling param"     
    else:
      sampling_params = [sampling_params] * len(inputs)
    
    #reset our accumulators
    if self.benchmark:
      self.start_times.clear()
      self.ttft.clear()
      self.first_token_times.clear()
      self.last_token_times.clear()
      self.itls.clear()
    
    self.request_ids = []
    self.results = {}
    #add prompts to waiting
    for input, sampling_param in zip(inputs, sampling_params):
      self.add_prompt(input, sampling_param)

    #clear all running and waiting sequences
    while not self.scheduler.is_finished():
      scheduled_sequences, is_prefill = self.scheduler.schedule()

      outputs = self.runner.run(scheduled_sequences, is_prefill)

      #get ttft for benchmark
      if self.benchmark:
        end_time = perf_counter()
        for sequence in scheduled_sequences:

          finished_prefill = (
              sequence.num_computed_tokens + 
              sequence.num_scheduled_tokens ==
              len(sequence)
          )

          produces_token = (
              not is_prefill or finished_prefill
          )

          if not produces_token: continue

          if sequence.seq_id not in self.first_token_times:
            self.first_token_times[sequence.seq_id] = end_time
            self.ttft[sequence.seq_id] = end_time - self.start_times[sequence.seq_id]
          
          else:
            self.itls.append(end_time - self.last_token_times[sequence.seq_id])
          
          self.last_token_times[sequence.seq_id] = end_time

      #postprocessing scheduled sequences
      self.scheduler.postprocess(scheduled_sequences, outputs, is_prefill)

      for sequence in scheduled_sequences:
        if sequence.status == SequenceStatus.FINISHED:
          output = self.tokenizer.decode(sequence.token_ids[sequence.num_prompt_tokens:], skip_special_tokens=True)
          self.results[sequence.seq_id] = output

    if self.benchmark:
      #get average itl and ttft
      average_ttft = sum(self.ttft.values()) / len(self.ttft)
      average_itl = sum(self.itls) / len(self.itls) if self.itls else None

      return average_ttft, average_itl

    return [self.results[seq] for seq in self.request_ids]

    