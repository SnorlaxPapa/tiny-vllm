class SamplingParams:
    def __init__(self, temperature: float = 1.0, max_tokens: int = 64, ignore_eos: bool = False):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos