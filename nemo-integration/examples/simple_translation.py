"""
Simple Translation Example

Demonstrates basic usage of the NeMo translator for Python to Rust translation.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.translation.translator import NeMoTranslator, TranslationConfig
from src.validation.validator import TranslationValidator, ValidationLevel


def example_simple_function():
    """Example: Translate a simple function."""
    print("=" * 60)
    print("Example 1: Simple Function Translation")
    print("=" * 60)

    python_code = """
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b
"""

    config = TranslationConfig(
        model_path="models/translator.nemo",  # Use mock in absence of real model
        gpu_enabled=False,  # Set to True for GPU acceleration
        validate_output=True
    )

    with NeMoTranslator(config) as translator:
        result = translator.translate_function(python_code)

        print("\nPython Code:")
        print(python_code)

        print("\nRust Code:")
        print(result.rust_code)

        print(f"\nConfidence: {result.confidence:.2f}")
        print(f"Processing Time: {result.processing_time_ms:.2f}ms")

        if result.imports:
            print(f"\nRequired Imports:")
            for imp in result.imports:
                print(f"  {imp}")


def example_with_error_handling():
    """Example: Translate function with exception handling."""
    print("\n" + "=" * 60)
    print("Example 2: Function with Error Handling")
    print("=" * 60)

    python_code = """
def divide(a: int, b: int) -> float:
    \"\"\"Divide two numbers.\"\"\"
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""

    config = TranslationConfig(
        model_path="models/translator.nemo",
        gpu_enabled=False,
        validate_output=True
    )

    with NeMoTranslator(config) as translator:
        result = translator.translate_function(python_code)

        print("\nPython Code:")
        print(python_code)

        print("\nRust Code:")
        print(result.rust_code)

        # Validate the translation
        validator = TranslationValidator(level=ValidationLevel.FULL)
        validation = validator.validate(result.rust_code, python_code)

        print(f"\nValidation Status: {validation.status.value}")
        print(f"Checks Passed: {len(validation.passed_checks)}")
        print(f"Checks Failed: {len(validation.failed_checks)}")

        if validation.get_errors():
            print("\nErrors:")
            for error in validation.get_errors():
                print(f"  - {error.message}")

        if validation.get_warnings():
            print("\nWarnings:")
            for warning in validation.get_warnings():
                print(f"  - {warning.message}")


def example_class_translation():
    """Example: Translate a Python class."""
    print("\n" + "=" * 60)
    print("Example 3: Class Translation")
    print("=" * 60)

    python_code = """
class Point:
    \"\"\"A 2D point.\"\"\"

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_from_origin(self) -> float:
        \"\"\"Calculate distance from origin.\"\"\"
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y})"
"""

    config = TranslationConfig(
        model_path="models/translator.nemo",
        gpu_enabled=False
    )

    with NeMoTranslator(config) as translator:
        result = translator.translate_class(python_code)

        print("\nPython Code:")
        print(python_code)

        print("\nRust Code:")
        print(result.rust_code)


def example_batch_translation():
    """Example: Batch translate multiple functions."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Translation")
    print("=" * 60)

    python_codes = [
        "def square(x: int) -> int:\n    return x * x",
        "def cube(x: int) -> int:\n    return x * x * x",
        "def is_even(n: int) -> bool:\n    return n % 2 == 0",
    ]

    config = TranslationConfig(
        model_path="models/translator.nemo",
        gpu_enabled=False,
        batch_size=3
    )

    with NeMoTranslator(config) as translator:
        results = translator.batch_translate(python_codes)

        for i, (python_code, result) in enumerate(zip(python_codes, results), 1):
            print(f"\n--- Function {i} ---")
            print(f"Python: {python_code}")
            print(f"Rust:   {result.rust_code}")
            print(f"Confidence: {result.confidence:.2f}")


def example_with_type_inference():
    """Example: Type inference for untyped Python code."""
    print("\n" + "=" * 60)
    print("Example 5: Type Inference")
    print("=" * 60)

    from src.mapping.type_mapper import TypeMapper

    python_code = """
def process_data(items):
    \"\"\"Process a list of items.\"\"\"
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result
"""

    # Infer types from usage
    mapper = TypeMapper()

    # Simulate type inference (in practice, this would use runtime analysis)
    inferred_types = {
        "items": "List[int]",
        "item": "int",
        "result": "List[int]",
        "__return__": "List[int]"
    }

    print("Inferred Types:")
    for name, type_str in inferred_types.items():
        rust_type = mapper.map_annotation(type_str)
        print(f"  {name}: {type_str} → {rust_type.name}")

    config = TranslationConfig(
        model_path="models/translator.nemo",
        gpu_enabled=False
    )

    with NeMoTranslator(config) as translator:
        # Provide type hints in context
        context = {"type_hints": inferred_types}
        result = translator.translate_function(python_code, context)

        print("\nPython Code:")
        print(python_code)

        print("\nRust Code (with inferred types):")
        print(result.rust_code)


def example_statistics():
    """Example: Translation statistics."""
    print("\n" + "=" * 60)
    print("Example 6: Translation Statistics")
    print("=" * 60)

    config = TranslationConfig(
        model_path="models/translator.nemo",
        gpu_enabled=False
    )

    translator = NeMoTranslator(config)
    translator.initialize()

    # Translate several functions
    test_codes = [
        "def f1(): return 1",
        "def f2(x): return x + 1",
        "def f3(a, b): return a * b",
        "def f4(lst): return sum(lst)",
    ]

    for code in test_codes:
        translator.translate_function(code)

    # Get statistics
    stats = translator.get_statistics()

    print("\nTranslation Statistics:")
    print(f"  Total Translations: {stats['translations']}")
    print(f"  Successes: {stats['successes']}")
    print(f"  Failures: {stats['failures']}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    print(f"  Average Time: {stats['avg_time_ms']:.2f}ms")

    translator.cleanup()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("NeMo Translation Examples")
    print("=" * 60)

    try:
        example_simple_function()
        example_with_error_handling()
        example_class_translation()
        example_batch_translation()
        example_with_type_inference()
        example_statistics()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
