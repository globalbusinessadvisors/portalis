"""Sample calculator module for assessment testing."""

from typing import List, Optional


class Calculator:
    """A simple calculator with history tracking."""

    def __init__(self):
        self.history: List[str] = []

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a: float, b: float) -> Optional[float]:
        """Divide a by b."""
        if b == 0:
            return None
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def get_history(self) -> List[str]:
        """Get calculation history."""
        return self.history.copy()

    def clear_history(self):
        """Clear calculation history."""
        self.history.clear()

    def __str__(self) -> str:
        """String representation."""
        return f"Calculator(operations={len(self.history)})"


@property
def version() -> str:
    """Get version string."""
    return "1.0.0"


def create_calculator() -> Calculator:
    """Factory function to create a calculator."""
    return Calculator()
