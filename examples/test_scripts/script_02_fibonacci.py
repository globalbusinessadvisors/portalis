"""
Test Script 2: Fibonacci (Recursive)
Target: ✅ MUST PASS

Tests recursion and conditional logic.
"""

def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def fib_iterative(n: int) -> int:
    """Calculate nth Fibonacci number iteratively."""
    if n <= 1:
        return n

    a = 0
    b = 1
    for i in range(2, n + 1):
        temp = a + b
        a = b
        b = temp

    return b
