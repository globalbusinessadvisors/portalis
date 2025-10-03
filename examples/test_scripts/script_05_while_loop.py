"""
Test Script 5: While Loop
Target: ✅ MUST PASS

Tests while loop constructs.
"""

def count_down(n: int) -> int:
    """Count down from n to 0."""
    while n > 0:
        n = n - 1
    return n

def sum_to_n(n: int) -> int:
    """Sum all numbers from 1 to n using while loop."""
    total = 0
    i = 1
    while i <= n:
        total = total + i
        i = i + 1
    return total

def power_of_two(n: int) -> int:
    """Calculate 2^n using while loop."""
    result = 1
    count = 0
    while count < n:
        result = result * 2
        count = count + 1
    return result
