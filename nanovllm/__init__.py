"""nano-vLLM-style public API; existing EngineCore imports remain supported."""


def __getattr__(name):
    if name == "LLM":
        from nanovllm.engine.engine import EngineCore
        return EngineCore
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams
        return SamplingParams
    raise AttributeError(name)
