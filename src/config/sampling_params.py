from dataclasses import dataclass

@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 512
    ignore_eos: bool = False

    # execute after initialization
    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
