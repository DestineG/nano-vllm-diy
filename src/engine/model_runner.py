import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoConfig

from src.engine.sequence import Sequence
from src.layers.sampler import Sampler
from src.utils.context import Context
from src.config.runner_config import RunnerConfig


class ModelRunner:

    def __init__(
        self,
        rank: int,
        event: Event | list[Event],
        modelClass = None,
        runner_cfg: RunnerConfig = None,
        model_cfg: AutoConfig = None,
    ):
        self.block_size = runner_cfg.kvcache_block_size
        self.enforce_eager = runner_cfg.enforce_eager
        self.world_size = runner_cfg.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(runner_cfg.dtype)
        torch.set_default_device("cuda")
        self.model = modelClass(model_cfg, tp=(rank, self.world_size))
        self.model.load(runner_cfg.model_path)
        self.sampler = Sampler()
        self.warmup_model(
            max_batched_seq_len=runner_cfg.max_batched_seq_len,
            max_seq_len=runner_cfg.max_seq_len,
            max_num_seqs=runner_cfg.max_num_seqs
        )
        self.num_kvcache_blocks = self.allocate_kv_cache(
            gpu_memory_utilization=runner_cfg.gpu_memory_utilization,
            num_kvcache_blocks=runner_cfg.num_kvcache_blocks,
            kvcache_block_size=runner_cfg.kvcache_block_size,
            num_key_value_heads=runner_cfg.num_key_value_heads,
            head_dim=runner_cfg.head_dim,
            num_hidden_layers=runner_cfg.num_hidden_layers,
            dtype=runner_cfg.dtype,
            tp_world_size=self.world_size,
        )
        if not self.enforce_eager:
            self.capture_cudagraph(
                max_num_seqs=runner_cfg.max_num_seqs,
                max_seq_len=runner_cfg.max_seq_len,
                model_dim=runner_cfg.model_dim,
            )
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="src", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="src")
                self.loop()

    def exit(self):
        print(f"RANK {self.rank} num_blocks={self.num_kvcache_blocks} exiting...")
        print(f"RANK {self.rank} exiting...")
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        print(f"RANK {self.rank} released shared memory.")
        if not self.enforce_eager:
            graphs = getattr(self, "graphs", None)
            graph_pool = getattr(self, "graph_pool", None)
            if graphs is not None:
                del self.graphs
            if graph_pool is not None:
                del self.graph_pool
        print(f"RANK {self.rank} released CUDA graphs.")
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(
        self,
        max_batched_seq_len: int=4096 * 4,
        max_seq_len: int=4096,
        max_num_seqs: int=512
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seq_len = min(max_batched_seq_len, max_seq_len)
        num_seqs = min(max_batched_seq_len // seq_len, max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs)
        torch.cuda.empty_cache()

    def allocate_kv_cache(
        self,
        gpu_memory_utilization: float = 0.9,
        num_kvcache_blocks: int | None = None,
        kvcache_block_size: int = 512,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        num_hidden_layers: int | None = None,
        dtype: torch.dtype = torch.float16,
        tp_world_size: int | None = None,
    ):
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        local_num_kv_heads = num_key_value_heads // tp_world_size
        block_bytes = 2 * num_hidden_layers * kvcache_block_size * local_num_kv_heads * head_dim * dtype.itemsize
        max_num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
        if num_kvcache_blocks is None or num_kvcache_blocks <= 0:
            num_kvcache_blocks = max_num_kvcache_blocks
        assert max_num_kvcache_blocks >= num_kvcache_blocks, f"Not enough GPU memory to allocate kv cache, required {num_kvcache_blocks * block_bytes / 2**30:.2f} GB but only {max_num_kvcache_blocks * block_bytes / 2**30:.2f} GB is available"
        kv_cache = torch.empty(2, num_hidden_layers, num_kvcache_blocks, kvcache_block_size, local_num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        return num_kvcache_blocks

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_seqs(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        context_lens = []
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            start = min(seq.num_cached_tokens, seqlen - 1)
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            # During chunk/prefix prefill, only keys up to `end` are guaranteed to exist in KV cache.
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            context_lens.append(seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        ctx = Context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        is_prefill = True
        if max_seqlen_q == 1:   # 纯 decode 场景走 graph.replay()
            is_prefill = False
        ctx.is_prefill = is_prefill
        return input_ids, positions, ctx, is_prefill

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, ctx: Context | None = None):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions, ctx), ctx)
        else:
            bs = input_ids.size(0)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = ctx.slot_mapping
            graph_vars["cu_seqlens_q"].zero_()
            graph_vars["cu_seqlens_q"][:bs + 1] = ctx.cu_seqlens_q
            graph_vars["cu_seqlens_k"].zero_()
            graph_vars["cu_seqlens_k"][:bs + 1] = ctx.cu_seqlens_k
            graph_vars["block_tables"].fill_(-1)
            graph_vars["block_tables"][:bs, :ctx.block_tables.size(1)] = ctx.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs], ctx)

    def run(self, seqs: list[Sequence]) -> list[int]:
        input_ids, positions, ctx, is_prefill = self.prepare_seqs(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill, ctx)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(
        self,
        max_num_seqs: int=512,
        max_seq_len: int=4096,
        model_dim: int=2048,
    ):
        max_bs = min(max_num_seqs, 512)
        max_num_blocks = (max_seq_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, model_dim)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            ctx = Context(
                is_prefill=True,
                cu_seqlens_q=cu_seqlens_q[:bs + 1],
                cu_seqlens_k=cu_seqlens_k[:bs + 1],
                max_seqlen_q=1,
                max_seqlen_k=max_seq_len,
                slot_mapping=slot_mapping[:bs],
                block_tables=block_tables[:bs]
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs], ctx)    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs], ctx)    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            outputs=outputs,
        )