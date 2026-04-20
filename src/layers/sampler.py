import torch
from torch import nn

class Sampler(nn.Module):
    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor
    ):
        # ParallelLMHead 的输出 logits 形状为 (num_seq, vocab_size)，temperatures 形状为 (num_seq, )，每个序列一个温度
        # logits: (num_seq, vocab_size)
        # temperatures: (num_seq, )
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
