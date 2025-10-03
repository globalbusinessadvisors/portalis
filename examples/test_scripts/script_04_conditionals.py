"""
Test Script 4: Conditionals
Target: ✅ MUST PASS

Tests if/elif/else statements.
"""

def max_of_two(a: int, b: int) -> int:
    """Return the maximum of two numbers."""
    if a > b:
        return a
    else:
        return b

def max_of_three(a: int, b: int, c: int) -> int:
    """Return the maximum of three numbers."""
    if a >= b and a >= c:
        return a
    elif b >= c:
        return b
    else:
        return c

def sign(n: int) -> int:
    """Return -1, 0, or 1 based on sign of n."""
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0
