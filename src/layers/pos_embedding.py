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
        x: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        # x: (batch_seq_len, num_heads, head_dim); positions: (batch_seq_len, )
        # cos_sin: (max_seq_len, 1, head_dim) -> (batch_seq_len, 1, head_dim)
        cos_sin = self.cos_sin[positions]
        x1, x2 = torch.chunk(x.float(), 2, dim=-1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos
        return torch.cat((x1_rotated, x2_rotated), dim=-1).to(x.dtype)