from functools import lru_cache
import torch
from torch import nn


@lru_cache(maxsize=1)
def build_rope_cos_sin(
    max_seq_len: int,
    head_dim: int,
    base: int = 10000
) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    position_ids = torch.arange(max_seq_len, dtype=torch.float)
    # (max_seq_len, ) @ (head_dim // 2, ) -> (max_seq_len, head_dim // 2)
    freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
    # (max_seq_len, head_dim // 2) -> (max_seq_len, head_dim) -> (max_seq_len, 1, head_dim)
    cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
    return cos_sin

class RoPE(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        head_dim: int,
        base: int = 1000000
    ) -> None:
        super().__init__()
        self.register_buffer("cos_sin", build_rope_cos_sin(max_seq_len, head_dim, base), persistent=False)

    @torch.compile
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (batch_seq_len, num_heads/num_kv_heads, head_dim)
        # positions: (batch_seq_len, )
        
        cos_sin = self.cos_sin[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        def rotate_tensor(t):
            # 转换为 float32 保证计算精度
            t1, t2 = torch.chunk(t.float(), 2, dim=-1)
            t1_rotated = t1 * cos - t2 * sin
            t2_rotated = t1 * sin + t2 * cos
            return torch.cat((t1_rotated, t2_rotated), dim=-1).to(t.dtype)

        q_rotated = rotate_tensor(q)
        k_rotated = rotate_tensor(k)
        return q_rotated, k_rotated