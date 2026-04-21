import os
import time
from random import randint, seed
from src.engine.llm_engine import LLMEngine as LLM
from src.models.qwen3 import Qwen3ForCausalLM
from src.config.sampling_params import SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, Qwen3ForCausalLM)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    print(llm.generate(["Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have?"], SamplingParams()))
    # t = time.time()
    # llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    # t = (time.time() - t)
    # total_tokens = sum(sp.max_tokens for sp in sampling_params)
    # throughput = total_tokens / t
    # print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
