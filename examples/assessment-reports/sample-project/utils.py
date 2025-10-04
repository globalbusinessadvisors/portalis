"""Utility functions for the calculator."""

from typing import List


def format_result(value: float) -> str:
    """Format a numeric result as a string."""
    return f"{value:.2f}"


def parse_expression(expr: str) -> List[str]:
    """Parse a mathematical expression into tokens."""
    tokens = []
    current = ""

    for char in expr:
        if char in "+-*/":
            if current:
                tokens.append(current)
                current = ""
            tokens.append(char)
        elif char.isspace():
            if current:
                tokens.append(current)
                current = ""
        else:
            current += char

    if current:
        tokens.append(current)

    return tokens


def validate_number(value: str) -> bool:
    """Check if a string represents a valid number."""
    try:
        float(value)
        return True
    except ValueError:
        return False
