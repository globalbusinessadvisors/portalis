"""
Test Script 8: GCD (Greatest Common Divisor)
Target: ✅ MUST PASS

Tests Euclidean algorithm with while loop.
"""

def gcd(a: int, b: int) -> int:
    """Calculate GCD using Euclidean algorithm."""
    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a

def lcm(a: int, b: int) -> int:
    """Calculate LCM using GCD."""
    return (a * b) // gcd(a, b)
