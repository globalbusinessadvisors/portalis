"""Test classes for class translation validation."""

class Calculator:
    """Simple calculator class."""

    def __init__(self, precision: int):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        return round(a + b, self.precision)

    def subtract(self, a: float, b: float) -> float:
        return round(a - b, self.precision)


class Counter:
    """Simple counter class."""

    def __init__(self):
        self.count = 0

    def increment(self) -> int:
        self.count = self.count + 1
        return self.count

    def get_count(self) -> int:
        return self.count

    def reset(self):
        self.count = 0


class Rectangle:
    """Rectangle with width and height."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def scale(self, factor: float):
        self.width = self.width * factor
        self.height = self.height * factor
