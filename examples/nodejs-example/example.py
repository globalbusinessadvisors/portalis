def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n: int) -> int:
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Test the functions
fib_result = fibonacci(10)
fact_result = factorial(5)
print(fib_result)
print(fact_result)
