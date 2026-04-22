from collections import deque
import xxhash
import numpy as np

from src.engine.sequence import Sequence
from src.utils.log import add_hit

class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        
        self.token_ids = []
        self.hash = -1
        self.cached = False
        
        self.ref_count = 0
    
    def update(
        self,
        hash: int | None = None,
        token_ids: list[int] | None = None,
        cached: bool | None = None
    ):
        if hash is not None:
            self.hash = hash
        if token_ids is not None:
            self.token_ids = token_ids
        if cached is not None:
            self.cached = cached
    
    def reset(self):
        self.token_ids = []
        self.hash = -1
        self.ref_count = 0
        self.cached = False
    
    def add_ref(self):
        self.ref_count += 1
    
    def remove_ref(self):
        self.ref_count -= 1
    
    def reset_ref(self):
        self.ref_count = 0
    

class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int
    ):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()
    
    def compute_hash(
        self,
        token_ids: list[int],
        prefix_hash: int=-1
    ):
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, byteorder="little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    def _allocate_block(self, block_id: int):
        block = self.blocks[block_id]
        # 如果这个块之前有旧的 hash，需要先从全局映射中移除旧的 mapping
        if block.hash != -1 and block.hash in self.hash_to_block_id:
            # 注意：只有当 hash 指向的确实是当前 block 时才删除
            if self.hash_to_block_id[block.hash] == block_id:
                del self.hash_to_block_id[block.hash]
                
        block.reset() # 只有在这里才真正清空
        block.add_ref()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block
    
    def _free_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count > 0, "Block is already free"
        block.remove_ref()
        if block.ref_count == 0:
            # block.reset()
            self.free_block_ids.append(block_id)
            self.used_block_ids.remove(block_id)

    def compute_nums_prefix_cache_token(self, seq: Sequence):
        '''计算 seq 的前缀中有多少个 token 是在 block cache 中的'''
        nums_prefix_cache_token = 0
        h = -1
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            if len(token_ids) < self.block_size:
                break
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if (block_id == -1                                      # hash 不存在
                or not self.blocks[block_id].cached                 # cache 未填充
                or self.blocks[block_id].token_ids != token_ids):   # token_ids 不匹配
                break
            nums_prefix_cache_token += self.block_size
        return nums_prefix_cache_token

    def can_allocate(self, seq: Sequence):
        '''判断是否有足够的 free block 来分配给 seq'''
        assert not seq.block_table, "Sequence already has a block table"
        num_cached_tokens = self.compute_nums_prefix_cache_token(seq)
        num_cached_blocks = num_cached_tokens // self.block_size
        num_required_new_blocks = seq.num_blocks - num_cached_blocks
        return len(self.free_block_ids) >= num_required_new_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table, "Sequence already has a block table"
        h = -1
        num_cache_hit = 0
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 只在完整 block 上计算 hash
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if (not cache_miss                                              # cache 未命中
                and (block_id == -1                                         # hash 不存在
                     or not self.blocks[block_id].cached                    # cache 未填充
                     or self.blocks[block_id].token_ids != token_ids)):     # token_ids 不匹配
                cache_miss = True
            # 非完整 block / 完整 block
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
                block.update(h, token_ids, cached=False)
                if h != -1:
                    self.hash_to_block_id[h] = block_id
            # 完整 block
            else:
                add_hit()
                num_cache_hit += self.block_size
                seq.num_cached_tokens += Sequence.block_size
                block = self.blocks[block_id]
                block.add_ref()
                if block_id in self.free_block_ids:
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        return num_cache_hit
    
    def deallocate(self, seq: Sequence):
        assert seq.block_table, "Sequence does not have a block table"
        for block_id in seq.block_table:
            self._free_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
    
    def can_append(self, seq: Sequence):
        '''在当前 block 已满的情况下，判断是否有足够的 free block 来存储新的 token'''
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
    
    def may_append(self, seq: Sequence):
        '''根据 seq 更新其 block_table, 分为三种情况:
        1. len(seq) % block_size == 1: 需要分配新的 block, 并将新的 block_id 加入 block_table
        2. len(seq) % block_size == 0: 需要更新最后一个 block 的状态(hash 和 token_ids)
        3. 其他情况: 不需要分配新的 block, 也不需要更新 block 的状态
        '''
        assert seq.block_table, "Sequence does not have a block table"
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # 仅做 block 的分配
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1, "Last block should be full"
            assert len(self.free_block_ids) > 0, "No free blocks available"
            block_id = self.free_block_ids[0]
            _ = self._allocate_block(block_id)
            block_table.append(block_id)
        # 仅做 block 状态的更新
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1, "Last block should not be full"
            token_ids = seq.block(seq.num_blocks - 1)
            prefix_hash = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix_hash)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        # 啥也不做
        else:
            assert last_block.hash == -1, "Last block should not be full"

    def update_block_cache_status(self, seq: Sequence, token_start, nums_token):
        '''
        token_start: 本次新生成的第一个 token 的索引 (0-based)
        nums_token: 本次操作新生成的 token 数量
        '''
        num_before = token_start 
        num_after = token_start + nums_token
        
        blocks_full_before = num_before // self.block_size
        blocks_full_after = num_after // self.block_size
        
        if blocks_full_after > blocks_full_before:
            for block_idx in range(blocks_full_before, blocks_full_after):
                if block_idx >= seq.num_blocks:
                    raise ValueError("block_idx out of range")
                block_id = seq.block_table[block_idx]
                block = self.blocks[block_id]
                block.update(cached=True)