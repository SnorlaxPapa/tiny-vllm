from transformers import AutoTokenizer
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.layers.sampler import Sampler

class EngineCore:

  def __init__(self, model_dir):
    self.runner = ModelRunner(model_dir) #allocate kv cache 
    config = self.runner.config
    self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config.eos = self.tokenizer.eos_token_id #get eos marker
    self.scheduler = Scheduler(config)
    self.request_ids = []
    self.results = {}


  def add_prompt(self, input: str, sampling_param: SamplingParams):
    token_ids = self.tokenizer.apply_chat_template(
      [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
       {"role": "user", "content": input}],
      tokenize=True,
      add_generation_prompt=True,
    )
    sequence = Sequence(token_ids["input_ids"], sampling_param)
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
    
    self.request_ids = []
    self.results = {}
    #add prompts to waiting
    for input, sampling_param in zip(inputs, sampling_params):
      self.add_prompt(input, sampling_param)

    #clear all running and waiting sequences
    while not self.scheduler.is_finished():
      scheduled_sequences, is_prefill = self.scheduler.schedule()

      outputs = self.runner.run(scheduled_sequences, is_prefill)

      #postprocessing scheduled sequences
      self.scheduler.postprocess(scheduled_sequences, outputs, is_prefill)

      for sequence in scheduled_sequences:
        if sequence.status == SequenceStatus.FINISHED:
          output = self.tokenizer.decode(sequence.token_ids[sequence.num_prompt_tokens:], skip_special_tokens=True)
          self.results[sequence.seq_id] = output


    return [self.results[seq] for seq in self.request_ids]

    