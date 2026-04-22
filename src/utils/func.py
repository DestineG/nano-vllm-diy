def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-indexed).

    Args:
        n: Index in Fibonacci sequence, must be >= 0.

    Returns:
        The Fibonacci number at index n.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
