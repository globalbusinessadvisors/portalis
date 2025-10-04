"""
Simple Calculator Module - Beta Test Project
Lines of Code: ~100
Complexity: Low
Purpose: Test basic Python to Rust translation
"""

from typing import Union, List
import math


class Calculator:
    """A simple calculator with basic arithmetic operations."""

    def __init__(self):
        self.history: List[str] = []

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a: float, b: float) -> Union[float, str]:
        """Divide a by b. Returns error message if b is zero."""
        if b == 0:
            return "Error: Division by zero"
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def power(self, base: float, exponent: float) -> float:
        """Calculate base raised to exponent."""
        result = base ** exponent
        self.history.append(f"{base} ^ {exponent} = {result}")
        return result

    def sqrt(self, n: float) -> Union[float, str]:
        """Calculate square root. Returns error for negative numbers."""
        if n < 0:
            return "Error: Cannot calculate square root of negative number"
        result = math.sqrt(n)
        self.history.append(f"√{n} = {result}")
        return result

    def get_history(self) -> List[str]:
        """Return calculation history."""
        return self.history.copy()

    def clear_history(self) -> None:
        """Clear calculation history."""
        self.history.clear()


def calculate_average(numbers: List[float]) -> float:
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def calculate_sum(numbers: List[float]) -> float:
    """Calculate the sum of a list of numbers."""
    return sum(numbers)


def find_min_max(numbers: List[float]) -> tuple:
    """Find minimum and maximum values in a list."""
    if not numbers:
        return (0.0, 0.0)
    return (min(numbers), max(numbers))


def main():
    """Demo the calculator functionality."""
    calc = Calculator()

    # Basic operations
    print("Addition: 10 + 5 =", calc.add(10, 5))
    print("Subtraction: 10 - 5 =", calc.subtract(10, 5))
    print("Multiplication: 10 * 5 =", calc.multiply(10, 5))
    print("Division: 10 / 5 =", calc.divide(10, 5))
    print("Power: 2 ^ 8 =", calc.power(2, 8))
    print("Square root: √16 =", calc.sqrt(16))

    # Error cases
    print("Division by zero:", calc.divide(10, 0))
    print("Square root of negative:", calc.sqrt(-4))

    # List operations
    numbers = [1.5, 2.5, 3.5, 4.5, 5.5]
    print(f"\nNumbers: {numbers}")
    print("Average:", calculate_average(numbers))
    print("Sum:", calculate_sum(numbers))
    print("Min/Max:", find_min_max(numbers))

    # History
    print("\nCalculation History:")
    for entry in calc.get_history():
        print(f"  {entry}")


if __name__ == "__main__":
    main()
