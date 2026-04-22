from collections import deque

from src.config.scheduler_cfg import SchedulerConfig
from src.engine.sequence import Sequence, SequenceStatus
from src.engine.block_manager import BlockManager
from src.utils.log import log_count

class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_seq_len = config.max_seq_len
        self.eos = config.eos
        self.chunked_prefill_size = config.chunked_prefill_size
        self.block_manager = BlockManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size
        )
        self.prefill: deque[Sequence] = deque()
        self.decode: deque[Sequence] = deque()
    
    def is_finished(self):
        return len(self.prefill) == 0 and len(self.decode) == 0
    
    def add_sequence(self, seq: Sequence):
        self.prefill.append(seq)
    
    def back_to_prefill(self, seq: Sequence):
        assert seq.status == SequenceStatus.DECODE, "Only sequences in DECODE status can be moved back to PREFILL"
        seq.status = SequenceStatus.PREFILL
        self.block_manager.deallocate(seq)
        self.prefill.appendleft(seq)
    
    def schedule(self):
        scheduled_seqs = []
        prefill_num_batched_tokens = 0
        decode_num_batched_tokens = 0
        max_num_chunk_prefill = 1
        num_chunk_prefill = 0

        # decode
        while self.decode and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.decode.popleft()
            while not self.block_manager.can_append(seq):
                if self.decode:
                    self.back_to_prefill(self.decode.pop())
                else:
                    self.back_to_prefill(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                decode_num_batched_tokens += 1
                scheduled_seqs.append(seq)
        self.decode.extendleft(reversed(scheduled_seqs))

        # prefill: don't support chunk prefill for now
        while self.prefill and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.prefill[0]

            # 计算 seq 本次调度的 token 数量
            _, call_back = self.block_manager.compute_num_prefix_cache_block(seq)
            call_back(allow_append=True)    # 更新 seq 的 prefixcache 以及 blockmanager 的 used/free list
            num_free_block = self.block_manager.get_num_free_block()

            remaining_tokens = self.max_num_batched_tokens - prefill_num_batched_tokens - decode_num_batched_tokens
            if remaining_tokens <= 0:
                break
            max_num_available_token = len(seq) - seq.num_cached_tokens
            if max_num_available_token <= 0:    # 该序列的所有 token 都已缓存，直接进入 decode 阶段，重算最后一个 token
                seq.num_cached_tokens -= 1
                self.prefill.popleft()
                self.decode.append(seq)
                continue
            num_token_limit = min(remaining_tokens, max_num_available_token)

            num_remaining_block_table_token = len(seq.block_table) * self.block_manager.block_size - seq.num_cached_tokens
            num_oom_limit = num_free_block * self.block_manager.block_size + num_remaining_block_table_token
            if num_oom_limit <= 0:    # 没有可用的 block 了，无法调度该序列
                break

            is_chunked_prefill = max_num_available_token > min(num_token_limit, num_oom_limit)
            if is_chunked_prefill:
                if num_chunk_prefill < max_num_chunk_prefill:
                    num_scheduled_tokens = min(num_token_limit, num_oom_limit, self.chunked_prefill_size)
                    num_chunk_prefill += 1
                    log_count(chunk_prefill=1)
                else:
                    break
            else:
                num_scheduled_tokens = max_num_available_token
            seq.num_scheduled_tokens = num_scheduled_tokens

            # 根据 num_scheduled_tokens 计算是否需要为 seq 分配新的 block，以及更新 seq 的 block_table
            min_num_need_block = (seq.num_cached_tokens + num_scheduled_tokens + self.block_manager.block_size - 1) // self.block_manager.block_size
            if min_num_need_block > len(seq.block_table):
                self.block_manager.allocate(seq, min_num_need_block - len(seq.block_table))

            # 更新
            prefill_num_batched_tokens += num_scheduled_tokens
            scheduled_seqs.append(self.prefill.popleft())
        
        assert scheduled_seqs, "At least one sequence should be scheduled in each step to make progress"
        return scheduled_seqs, prefill_num_batched_tokens, decode_num_batched_tokens

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int]
    ):
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            self.block_manager.update_block_cache_status(
                seq,
                seq.num_cached_tokens - seq.num_scheduled_tokens,
                seq.num_scheduled_tokens
            )
            seq.num_scheduled_tokens = 0
            seq.append(token_id)

            if (seq.status == SequenceStatus.PREFILL
                and seq.num_cached_tokens == seq.num_tokens - 1
            ):
                seq.status = SequenceStatus.DECODE
                self.decode.append(seq)

            if (not seq.ignore_eos and token_id == self.eos) or seq.num_generated_token_ids == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.decode.remove(seq)