from dataclasses import dataclass

from torch import Tensor

@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: Tensor | None = None
    cu_seqlens_k: Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Tensor | None = None
    context_lens: Tensor | None = None
    block_tables: Tensor | None = None
