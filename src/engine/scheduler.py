from collections import deque

from src.config.scheduler_cfg import SchedulerConfig
from src.engine.sequence import Sequence, SequenceStatus
from src.engine.block_manager import BlockManager

class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            num_blocks=config.num_blocks,
            block_size=config.block_size
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
        num_batched_tokens = 0

        # prefill: don't support chunk prefill for now
        while self.prefill and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.prefill[0]
            num_prefix_cache_tokens =  self.block_manager.compute_nums_prefix_cache_token(seq)
            num_uncached_tokens = seq.num_tokens - num_prefix_cache_tokens
            remaining_tokens = self.max_num_batched_tokens - num_batched_tokens
            # 未 cache 的 token 数超过剩余的 batch token 数
            # or 已经没有足够的 free block 来存储未 cache 的 token
            if num_uncached_tokens > remaining_tokens or not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            seq.num_scheduled_tokens = num_uncached_tokens
            num_batched_tokens += num_uncached_tokens
            scheduled_seqs.append(self.prefill.popleft())
        if scheduled_seqs:
            return scheduled_seqs, True
        
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
                scheduled_seqs.append(seq)
        assert scheduled_seqs, "At least one sequence should be scheduled in decode stage"
        self.decode.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool
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

            if is_prefill:
                assert seq.num_cached_tokens == seq.num_prompt_tokens, "After prefill, all prompt tokens should be cached"
                seq.status = SequenceStatus.DECODE
                self.decode.append(seq)

            if (not seq.ignore_eos and token_id == self.eos) or seq.num_generated_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.decode.remove(seq)