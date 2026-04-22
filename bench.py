import os
import time
from random import randint, seed
from src.engine.llm_engine import LLMEngine as LLM
from src.models.qwen3 import Qwen3ForCausalLM
from src.config.sampling_params import SamplingParams
# from vllm import LLM, SamplingParams
from src.utils.log import reset_hit, get_hit

def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, Qwen3ForCausalLM)

    long_system_prompt = """
    You are a highly advanced logical reasoning engine. Your goal is to help users solve complex riddles, 
    mathematical problems, and programming challenges. 

    Guidelines:
    1. Always analyze the problem step-by-step using "Chain of Thought" (CoT).
    2. If a question is a trick question, identify the logical fallacy before answering.
    3. Use formal language and ensure all physical units (meters, liters, etc.) are double-checked.
    4. Be concise but thorough.

    Background Knowledge:
    - Earth's gravity: 9.81 m/s^2.
    - The speed of light: 299,792,458 m/s.
    - Volume of a rectangular prism: V = length * width * height.
    - A hole, by definition, contains no material.
    """
    short_messages = [
        {"role": "system", "content": long_system_prompt},
        {"role": "user", "content": "Can you explain the physics of excavation in detail?"},
    ]
    messages = [
        {"role": "system", "content": long_system_prompt},
        # 第一轮
        {"role": "user", "content": "Can you explain the physics of excavation in detail?"},
        {"role": "assistant", "content": "Excavation involves removing earth to create a cavity. The volume of the removed soil, often called 'spoil', is typically 20-30% greater than the volume of the hole itself due to the 'bulking factor' where the soil loosens and air is introduced."},
        # 第二轮
        {"role": "user", "content": "What are the safety requirements for a 2-meter deep hole?"},
        {"role": "assistant", "content": "For any excavation deeper than 1.5 meters, protective systems such as shoring, shielding (trench boxes), or sloping are required to prevent cave-ins. Professional engineering oversight is often necessary at 2 meters."},
        # 第三轮：
        {"role": "user", "content": "Now, back to my logic puzzle: If I dig that 2m x 3m x 4m hole, how much dirt is left inside?"}
    ]
    prompt = llm.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    reset_hit()
    print(prompt)
    print(llm.generate([prompt], SamplingParams(temperature=0.2, max_tokens=1024))[0]["text"])
    print(f"Prefix cache hit: {get_hit()}")
    print(llm.generate([prompt], SamplingParams(temperature=0.2, max_tokens=1024))[0]["text"])
    print(f"Prefix cache hit: {get_hit()}")


    # prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # # uncomment the following line for vllm
    # # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]
    # t = time.time()
    # reset_hit()
    # llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    # t = (time.time() - t)
    # total_tokens = sum(sp.max_tokens for sp in sampling_params)
    # throughput = total_tokens / t
    # print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    # print(f"Prefix cache hit: {get_hit()}")


if __name__ == "__main__":
    main()
