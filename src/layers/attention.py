import torch
from torch import nn
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


from src.utils.context import Context

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
    idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + idx)
    if slot_id == -1: return
    cache_offsets = slot_id * D + tl.arange(0, D)

    # 获取当前线程负责处理的 key/value
    offsets = idx * key_stride + tl.arange(0, D)
    key = tl.load(key_ptr + offsets)
    value = tl.load(value_ptr + offsets)

    # 将 key/value 存储到对应的 KV cache 插槽中
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    # key/value: (batch_seq_len, num_kv_heads, head_dim)
    # k_cache/v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    # slot_mapping: (batch_seq_len, )

    batch_seq_len, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim

    # 内存布局检查
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D

    # 插槽数量检查
    assert slot_mapping.numel() == batch_seq_len

    # 插槽 ID 检查，确保所有 slot_id 都在合法范围内
    # assert slot_mapping.min() >= 0 and slot_mapping.max() < k_cache.shape[0]

    store_kvcache_kernel[(batch_seq_len,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        scale: float = None
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: Context
    ):
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, ctx.slot_mapping)
        # TODO: will support chunk prefill in the future
        if ctx.is_prefill:
            # q: (batch_seq_len, num_heads, head_dim)
            attn_output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=ctx.cu_seqlens_q,
                cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=None
            )
        else:
            # q: (num_seq, num_heads, head_dim) -> (num_seq, 1, num_heads, head_dim)
            attn_output = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=ctx.context_lens,
                block_table=ctx.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
        return attn_output