"""
Test Script 7: Comparisons
Target: ✅ MUST PASS

Tests comparison and logical operators.
"""

def is_even(n: int) -> bool:
    """Check if number is even."""
    return n % 2 == 0

def is_positive(n: int) -> bool:
    """Check if number is positive."""
    return n > 0

def in_range(n: int, low: int, high: int) -> bool:
    """Check if n is between low and high (inclusive)."""
    return n >= low and n <= high

def is_valid(n: int) -> bool:
    """Check if n is positive and even."""
    return n > 0 and n % 2 == 0
