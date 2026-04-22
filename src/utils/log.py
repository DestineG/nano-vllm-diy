num_prefix_cache_hit=0

def add_hit():
    global num_prefix_cache_hit
    num_prefix_cache_hit += 1

def get_hit():
    return num_prefix_cache_hit

def reset_hit():
    global num_prefix_cache_hit
    num_prefix_cache_hit = 0