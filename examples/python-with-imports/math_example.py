import math

def calculate_circle(radius: float) -> float:
    area = math.pi * radius * radius
    return area

result = calculate_circle(5.0)
print(result)
