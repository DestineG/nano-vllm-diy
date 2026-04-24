from collections import deque
import xxhash
import numpy as np

from src.engine.sequence import Sequence
from src.utils.log import log_count

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
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()

    def get_num_free_block(self):
        return len(self.free_block_ids)

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
        block.reset()
        block.add_ref()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block
    
    def _free_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count > 0, "Block is already free"
        block.remove_ref()
        if block.ref_count == 0:
            self.free_block_ids.append(block_id)
            self.used_block_ids.remove(block_id)

    def compute_num_prefix_cache_block(self, seq: Sequence):
        """
        计算该序列在当前全局 Block Cache 下理论上能命中的前缀块数量

        注意：
        1. 该函数本身是只读的，不会修改 seq 状态或任何全局引用计数
        2. 它不依赖 block_table 的当前长度，仅基于 token_ids 的 Hash 进行匹配
        3. 返回一个元组 (total_hits, update_fn)。调用 update_fn() 才会真正应用物理块的替换/追加
        4. update_fn 会返回 undo_fn；若本轮调度未提交该序列，可调用 undo_fn 回滚 update_fn 的副作用
        """
        start_block = seq.num_cached_tokens // self.block_size

        if start_block > 0:
            assert start_block <= len(seq.block_table), "Start block should not exceed current block table length"
            h = self.blocks[seq.block_table[start_block - 1]].hash
        else:
            h = -1
        new_hitted_ids = []
        for i in range(start_block, seq.num_blocks):
            token_ids = seq.block(i)
            if len(token_ids) < self.block_size: break
            
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)

            if (block_id == -1
                or not self.blocks[block_id].cached 
                or self.blocks[block_id].token_ids != token_ids):
                break
            new_hitted_ids.append(block_id)

        total_hits = start_block + len(new_hitted_ids)
        log_count(prefix_cache=len(new_hitted_ids))

        def update_prefix_cache_block(allow_append=False):
            prev_cached_tokens = seq.num_cached_tokens
            undo_records = []
            for idx, new_id in enumerate(new_hitted_ids):
                table_idx = start_block + idx
                
                if table_idx < len(seq.block_table):
                    old_id = seq.block_table[table_idx]
                    if old_id != new_id:
                        self._free_block(old_id)
                        old_became_free = old_id in self.free_block_ids
                        self.blocks[new_id].add_ref()
                        new_was_free = new_id in self.free_block_ids
                        seq.block_table[table_idx] = new_id
                        if new_was_free:
                            self.free_block_ids.remove(new_id)
                            self.used_block_ids.add(new_id)
                        undo_records.append(("replace", table_idx, old_id, new_id, old_became_free, new_was_free))
                elif allow_append:
                    self.blocks[new_id].add_ref()
                    new_was_free = new_id in self.free_block_ids
                    seq.block_table.append(new_id)
                    if new_was_free:
                        self.free_block_ids.remove(new_id)
                        self.used_block_ids.add(new_id)
                    undo_records.append(("append", new_id, new_was_free))
                else:
                    raise ValueError("Cannot append hit block without allow_append=True")
            seq.num_cached_tokens = total_hits * self.block_size
            
            def undo_prefix_cache_block_update():
                # TODO: Current approach mutates global state and rolls it back on failed scheduling.
                #       Consider migrating to a virtual-state planning path to avoid apply+undo churn.
                for record in reversed(undo_records):
                    if record[0] == "replace":
                        _, table_idx, old_id, new_id, old_became_free, _ = record
                        seq.block_table[table_idx] = old_id
                        self._free_block(new_id)
                        self.blocks[old_id].add_ref()
                        if old_became_free:
                            self.free_block_ids.remove(old_id)
                            self.used_block_ids.add(old_id)
                    else:
                        _, new_id, _ = record
                        assert seq.block_table and seq.block_table[-1] == new_id
                        seq.block_table.pop()
                        self._free_block(new_id)
                seq.num_cached_tokens = prev_cached_tokens

            return undo_prefix_cache_block_update

        return total_hits, update_prefix_cache_block

    def allocate(self, seq: Sequence, num_to_allocate: int):
        """
        仅负责从 free_block_ids 中划拨指定数量的 block 并初始化

        调用者需确保 num_to_allocate 是排除 Cache Hit 后的净需求
        """
        assert num_to_allocate <= len(self.free_block_ids), "Not enough free blocks"

        h = -1 if not seq.block_table else self.blocks[seq.block_table[-1]].hash
        for _ in range(num_to_allocate):
            token_ids = seq.block(len(seq.block_table))
            
            # hash 计算
            is_full = len(token_ids) == self.block_size
            h = self.compute_hash(token_ids, h) if is_full else -1

            # block 分配与初始化
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            block.update(h, token_ids, cached=False)

            if h != -1:
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
        return True
    
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