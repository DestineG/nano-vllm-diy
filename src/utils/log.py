num_prefix_cache_hit=0
num_chunk_prefill = 0

def log_count(prefix_cache: int | None = None, chunk_prefill: int | None = None):
    if prefix_cache is not None:
        global num_prefix_cache_hit
        num_prefix_cache_hit += prefix_cache
    if chunk_prefill is not None:
        global num_chunk_prefill
        num_chunk_prefill += chunk_prefill

def reset_log_count():
    global num_prefix_cache_hit, num_chunk_prefill
    num_prefix_cache_hit = 0
    num_chunk_prefill = 0

def print_log_count():
    print(f"Prefix Cache Hit: {num_prefix_cache_hit}, Chunk Prefill: {num_chunk_prefill}")