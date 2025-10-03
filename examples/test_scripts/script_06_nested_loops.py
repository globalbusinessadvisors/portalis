"""
Test Script 6: Nested Loops
Target: ✅ MUST PASS

Tests nested for loops.
"""

def sum_range(start: int, end: int) -> int:
    """Sum numbers in a range."""
    total = 0
    for i in range(start, end + 1):
        total = total + i
    return total

def multiply_range(start: int, end: int) -> int:
    """Multiply numbers in a range."""
    result = 1
    for i in range(start, end + 1):
        result = result * i
    return result
