"""
Test Script 3: Factorial (Loop)
Target: ✅ MUST PASS

Tests for loops and accumulation pattern.
"""

def factorial(n: int) -> int:
    """Calculate factorial using a loop."""
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result

def factorial_recursive(n: int) -> int:
    """Calculate factorial recursively."""
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)
