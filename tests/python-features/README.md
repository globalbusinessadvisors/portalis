# Python Language Feature Test Suite

This directory contains test cases for validating Portalis translation coverage of all 527 Python 3.12 language features cataloged in `PYTHON_LANGUAGE_FEATURES.md`.

## Directory Structure

Tests are organized into 16 categories matching the feature catalog:

```
tests/python-features/
├── basic-syntax/       # 52 features: literals, assignments, unpacking
├── operators/          # 48 features: arithmetic, comparison, bitwise, logical
├── control-flow/       # 28 features: if/elif/else, for, while, match/case
├── data-structures/    # 43 features: lists, dicts, sets, tuples, comprehensions
├── functions/          # 48 features: def, lambda, args, decorators, closures
├── classes/            # 65 features: OOP, inheritance, properties, slots
├── modules/            # 22 features: import, __init__, packages
├── exceptions/         # 18 features: try/except, raise, custom exceptions
├── context-managers/   # 12 features: with statement, __enter__/__exit__
├── iterators/          # 28 features: __iter__, __next__, generators, yield
├── async-await/        # 32 features: async/await, asyncio, async generators
├── type-hints/         # 38 features: PEP 484, typing module, generics
├── metaclasses/        # 22 features: __new__, __init_subclass__, class creation
├── descriptors/        # 16 features: __get__, __set__, properties
├── builtins/           # 71 features: built-in functions (abs, map, filter, etc.)
└── magic-methods/      # 84 features: __dunder__ methods
```

## Test File Naming Convention

Each test file follows the pattern:
- **Format**: `test_<feature_name>.py`
- **Example**: `test_simple_assignment.py`, `test_list_comprehension.py`

## Test Structure

Each test file contains:
1. **Python source** - Original Python code demonstrating the feature
2. **Expected Rust output** - Target Rust code after translation
3. **Test cases** - Pytest-based validation
4. **Complexity metadata** - Low/Medium/High/Very High rating

### Example Test Template

```python
"""
Feature: Simple Assignment
Category: Basic Syntax
Complexity: Low
Status: Not Implemented
"""

import pytest
from portalis.translator import translate_python_to_rust

# Python source code
PYTHON_SOURCE = '''
x = 42
y = "hello"
z = 3.14
'''

# Expected Rust output
EXPECTED_RUST = '''
let x: i32 = 42;
let y: &str = "hello";
let z: f64 = 3.14;
'''

def test_simple_assignment_translation():
    """Test translation of simple variable assignments."""
    result = translate_python_to_rust(PYTHON_SOURCE)
    assert result == EXPECTED_RUST

def test_simple_assignment_execution():
    """Test execution of translated Rust code."""
    rust_code = translate_python_to_rust(PYTHON_SOURCE)
    # Compile and run Rust code
    # Assert output matches expected behavior
    pass
```

## Complexity Distribution

| Complexity | Count | Percentage | Description |
|------------|-------|------------|-------------|
| Low        | 241   | 45.7%      | Direct Rust equivalents exist |
| Medium     | 159   | 30.2%      | Adaptation required, feasible |
| High       | 91    | 17.3%      | Significant effort needed |
| Very High  | 36    | 6.8%       | Extremely difficult/impossible |

## Test Execution

### Run all tests
```bash
pytest tests/python-features/ -v
```

### Run specific category
```bash
pytest tests/python-features/basic-syntax/ -v
```

### Run by complexity level
```bash
pytest tests/python-features/ -v -m "complexity_low"
```

### Generate coverage report
```bash
pytest tests/python-features/ --cov=portalis.translator --cov-report=html
```

## Test Status Tracking

Tests are marked with status markers:
- `@pytest.mark.not_implemented` - Feature not yet translated
- `@pytest.mark.implemented` - Feature translated, tests passing
- `@pytest.mark.partial` - Feature partially implemented
- `@pytest.mark.unsupported` - Feature cannot be translated (eval, exec, etc.)

### Check implementation progress
```bash
pytest tests/python-features/ --collect-only | grep -c "implemented"
```

## Integration with CI/CD

These tests are integrated into the Portalis CI/CD pipeline:
1. **PR validation** - All new translations must pass relevant tests
2. **Coverage gating** - PRs must maintain/improve coverage percentage
3. **Regression testing** - Full suite runs on main branch commits
4. **Performance benchmarks** - Translation speed tracked per category

## Contributing

When adding a new Python feature:
1. Document it in `PYTHON_LANGUAGE_FEATURES.md`
2. Create corresponding test file in appropriate category
3. Mark test as `@pytest.mark.not_implemented`
4. Implement translator support
5. Update test to `@pytest.mark.implemented`
6. Ensure test passes

## Phase 1 Goals

**Week 1-2 Target**:
- Create 527 test file stubs (one per feature)
- Implement translation for 50 Low complexity features
- Achieve 10% coverage of total feature set

**Week 3-4 Target**:
- Implement 100 Low + Medium complexity features
- Achieve 25% coverage

See `PHASE_1_WEEK_1_PLAN.md` for detailed schedule.
