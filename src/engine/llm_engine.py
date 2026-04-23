import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch

from src.config.runner_config import RunnerConfig
from src.config.model_config import get_model_config
from src.config.scheduler_cfg import SchedulerConfig
from src.config.sampling_params import SamplingParams
from src.engine.sequence import Sequence
from src.engine.scheduler import Scheduler
from src.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model_name_or_path, modelClass):
        model_config = get_model_config(model_name_or_path)
        dtype = model_config.dtype if hasattr(model_config, "dtype") else torch.float16
        max_seq_len = model_config.max_position_embeddings
        num_key_value_heads = model_config.num_key_value_heads
        head_dim = model_config.head_dim if hasattr(model_config, "head_dim") else model_config.hidden_size // model_config.num_key_value_heads
        num_hidden_layers = model_config.num_hidden_layers
        model_dim = model_config.hidden_size

        runner_config = RunnerConfig(
            model_path=model_name_or_path,
            dtype=dtype,
            max_seq_len=max_seq_len,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            num_hidden_layers=num_hidden_layers,
            model_dim=model_dim
        )
        self.max_seq_len = runner_config.max_seq_len

        Sequence.block_size = runner_config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, runner_config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(i, event, modelClass, runner_config, model_config))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(0, self.events, modelClass, runner_config, model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(runner_config.model_path, use_fast=True)
        scheduler_config = SchedulerConfig(
            max_num_seqs=runner_config.max_num_seqs,
            max_num_batched_tokens=runner_config.max_batched_seq_len,
            max_seq_len=runner_config.max_seq_len,
            eos=self.tokenizer.eos_token_id,
            num_kvcache_blocks=self.model_runner.num_kvcache_blocks,
            kvcache_block_size=runner_config.kvcache_block_size,
            chunked_prefill_size=runner_config.chunked_prefill_size
        )
        self.scheduler = Scheduler(scheduler_config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        prompt_len = len(prompt)
        max_new_tokens = sampling_params.max_tokens
        block_size = self.scheduler.block_manager.block_size
        total_num_blocks = self.scheduler.block_manager.num_blocks
        
        # 检查 Prompt 是否超过了物理 KV Cache 的承载能力
        needed_prompt_blocks = (prompt_len + block_size - 1) // block_size
        if needed_prompt_blocks > total_num_blocks:
            raise ValueError(
                f"Prompt is too long ({prompt_len} tokens). "
                f"It requires {needed_prompt_blocks} blocks, but the system only has {total_num_blocks} blocks. "
                f"Please reduce the prompt length or increase num_kvcache_blocks."
            )

        # 检查 Prompt + 生成长度 是否超过了系统总容量
        total_expected_tokens = prompt_len + max_new_tokens
        needed_total_blocks = (total_expected_tokens + block_size - 1) // block_size
        if needed_total_blocks > total_num_blocks:
            raise ValueError(
                f"The request (prompt {prompt_len} + max_tokens {max_new_tokens}) exceeds system capacity. "
                f"Required blocks: {needed_total_blocks}, Available blocks: {total_num_blocks}. "
                f"Try reducing sampling_params.max_tokens."
            )

        # 模型支持的最大长度
        if total_expected_tokens > self.max_seq_len:
             raise ValueError(
                f"Requested total length {total_expected_tokens} exceeds the model's "
                f"maximum sequence length configuration ({self.max_seq_len})."
            )

        seq = Sequence(prompt, sampling_params)
        self.scheduler.add_sequence(seq)

    def step(self):
        seqs, prefill_num_tokens, decode_num_tokens = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.generated_token_ids) for seq in seqs if seq.is_finished]
        return outputs, prefill_num_tokens, decode_num_tokens
    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, prefill_num_tokens, decode_num_tokens = self.step()
            if prefill_num_tokens > 0:
                prefill_throughput = prefill_num_tokens / (perf_counter() - t)
            if decode_num_tokens > 0:
                decode_throughput = decode_num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                if token_ids and token_ids[-1] == self.tokenizer.eos_token_id:
                    token_ids = token_ids[:-1]
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
