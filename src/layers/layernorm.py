import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    @torch.compile
    def normal_forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        float_x = x.float()
        var = float_x.square().mean(-1, keepdim=True) + self.eps
        norm_x = float_x * var.rsqrt()
        return (self.weight * norm_x).to(input_dtype)

    @torch.compile
    def residual_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor
    ) -> torch.Tensor:
        input_dtype = x.dtype
        float_x = x.float() + residual.float()
        var = float_x.square().mean(-1, keepdim=True) + self.eps
        norm_x = float_x * var.rsqrt()
        return (self.weight * norm_x).to(input_dtype), float_x.to(input_dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor=None
    ) -> torch.Tensor:
        # x: (batch_seq_len, num_heads, head_dim)
        if residual is None:
            return self.normal_forward(x)
        else:
            return self.residual_forward(x, residual)