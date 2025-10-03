"""Advanced math operations."""
from math.basic import add

def multiply(a: int, b: int) -> int:
    return a * b

def sum_and_multiply(a: int, b: int, c: int) -> int:
    # Uses add from basic module
    total = add(a, b)
    return multiply(total, c)
