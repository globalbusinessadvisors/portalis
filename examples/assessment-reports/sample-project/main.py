"""Main entry point for calculator application."""

import sys
from calculator import Calculator, create_calculator
from utils import format_result, parse_expression


def main():
    """Run the calculator application."""
    calc = create_calculator()

    print("Simple Calculator")
    print("=" * 40)

    while True:
        try:
            expr = input("\nEnter expression (or 'q' to quit): ")

            if expr.lower() == 'q':
                break

            tokens = parse_expression(expr)

            if len(tokens) != 3:
                print("Error: Invalid expression format")
                continue

            a = float(tokens[0])
            op = tokens[1]
            b = float(tokens[2])

            if op == '+':
                result = calc.add(int(a), int(b))
            elif op == '-':
                result = calc.subtract(int(a), int(b))
            elif op == '*':
                result = calc.multiply(int(a), int(b))
            elif op == '/':
                result = calc.divide(a, b)
                if result is None:
                    print("Error: Division by zero")
                    continue
            else:
                print(f"Error: Unknown operator {op}")
                continue

            print(f"Result: {format_result(result)}")

        except ValueError:
            print("Error: Invalid number")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

    # Show history
    print("\nCalculation History:")
    for entry in calc.get_history():
        print(f"  {entry}")


if __name__ == "__main__":
    main()
