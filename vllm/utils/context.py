import torch

class Context:
    def __init__(
        self,
        is_prefill: bool = False,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        slot_mapping: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
    ):
        self.is_prefill = is_prefill
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
    

_CONTEXT = Context()

def get_context():
    return _CONTEXT


def set_context(
    is_prefill: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()