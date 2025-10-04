# Simple Calculator - Beta Test Project

**Project Type**: Simple
**Lines of Code**: ~100 LOC
**Complexity**: Low
**Purpose**: Test basic Python to Rust translation

---

## Overview

This is a simple calculator module designed to test fundamental Python-to-Rust translation capabilities. It demonstrates:

- Basic class definition with methods
- Type hints and annotations
- Error handling
- List operations
- Simple data structures

---

## Features Tested

### Python Features
- [x] Class definition with `__init__`
- [x] Instance methods
- [x] Type hints (float, str, List, Union, tuple)
- [x] String formatting
- [x] List operations (append, copy, clear)
- [x] Mathematical operations (+, -, *, /, **, sqrt)
- [x] Conditional logic (if/else)
- [x] Built-in functions (sum, min, max, len)
- [x] Docstrings

### Expected Translation
- Calculator class → Rust struct with impl
- Type hints → Rust type system
- Lists → Vec<T>
- Union types → Result<T, E> or enum
- Error messages → Result with error variants

---

## How to Use

### Run with Python
```bash
python calculator.py
```

### Expected Output
```
Addition: 10 + 5 = 15.0
Subtraction: 10 - 5 = 5.0
Multiplication: 10 * 5 = 50.0
Division: 10 / 5 = 2.0
Power: 2 ^ 8 = 256.0
Square root: √16 = 4.0
Division by zero: Error: Division by zero
Square root of negative: Error: Cannot calculate square root of negative number

Numbers: [1.5, 2.5, 3.5, 4.5, 5.5]
Average: 3.5
Sum: 17.5
Min/Max: (1.5, 5.5)

Calculation History:
  10.0 + 5.0 = 15.0
  10.0 - 5.0 = 5.0
  10.0 * 5.0 = 50.0
  10.0 / 5.0 = 2.0
  2.0 ^ 8.0 = 256.0
  √16.0 = 4.0
```

---

## Translation Steps

### Step 1: Translate to Rust
```bash
portalis translate calculator.py --output calculator.rs
```

### Step 2: Build to WASM
```bash
portalis build calculator.rs --target wasm32-wasi
```

### Step 3: Validate
```bash
portalis validate calculator.wasm
```

### Step 4: Run
```bash
portalis run calculator.wasm
```

---

## Expected Translation Challenges

### Easy (Should Work)
- [x] Basic arithmetic operations
- [x] Class to struct conversion
- [x] Simple type hints
- [x] String formatting
- [x] List operations

### Medium (May Need Adjustment)
- [ ] Union types (Union[float, str])
- [ ] Error handling patterns
- [ ] Math library functions
- [ ] Tuple returns

### Advanced (Unlikely First Pass)
- [ ] Dynamic string formatting
- [ ] Complex error messages

---

## Success Criteria

**Translation Success**: ✓ if Rust code compiles
**Build Success**: ✓ if WASM builds without errors
**Runtime Success**: ✓ if output matches Python
**Performance**: Should be 2-5x faster than Python

---

## Test Cases

### Test 1: Basic Operations
```python
calc = Calculator()
assert calc.add(5, 3) == 8
assert calc.subtract(5, 3) == 2
assert calc.multiply(5, 3) == 15
assert calc.divide(6, 3) == 2
```

### Test 2: Error Handling
```python
assert calc.divide(5, 0) == "Error: Division by zero"
assert calc.sqrt(-4) == "Error: Cannot calculate square root of negative number"
```

### Test 3: List Operations
```python
assert calculate_average([1, 2, 3, 4, 5]) == 3.0
assert calculate_sum([1, 2, 3]) == 6.0
assert find_min_max([1, 5, 3]) == (1, 5)
```

---

## Benchmarking

### Expected Performance
- Python execution: ~5-10ms
- WASM execution: ~2-3ms
- Speedup: 2-3x

### Benchmark Command
```bash
portalis benchmark calculator.py
```

---

## Notes for Beta Testers

1. This is an intentionally simple project to validate basic translation
2. Focus on translation accuracy over performance
3. Report any type inference issues
4. Note any manual adjustments needed
5. Compare Rust output quality with expectations

---

## Deliverables

After translation, you should have:
- [x] calculator.rs (Rust source)
- [x] calculator.wasm (WASM binary)
- [x] Translation report
- [x] Performance comparison
- [x] Test results

---

## Feedback

Report issues:
- Translation failures: #beta-technical
- Type inference problems: #beta-technical
- Performance issues: #beta-performance
- Documentation gaps: #beta-general
