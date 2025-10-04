# Data Processing Library - Medium Complexity Beta Project

**Project Type**: Medium
**Lines of Code**: ~500 LOC
**Complexity**: Medium
**Purpose**: Test intermediate Python features and patterns

---

## Overview

This data processing library tests more advanced Python-to-Rust translation capabilities including:

- Dataclasses and enums
- Generic types and type variables
- Lambda functions and higher-order functions
- JSON serialization
- Statistical operations
- Class methods and static methods

---

## Features Tested

### Core Python Features
- [x] Dataclasses (@dataclass)
- [x] Enums (Enum class)
- [x] Generic types (Generic[T], TypeVar)
- [x] Type hints (Optional, Callable, Dict, List)
- [x] Lambda functions
- [x] List comprehensions
- [x] Dictionary comprehensions
- [x] Class methods (@classmethod)
- [x] Static methods (@staticmethod)
- [x] Property decorators

### Advanced Features
- [x] JSON serialization/deserialization
- [x] Statistics module functions
- [x] Higher-order functions (filter with callbacks)
- [x] Method chaining
- [x] Complex data structures
- [x] Error validation patterns

### Expected Translation Challenges
- [ ] Dataclass to Rust struct with derive macros
- [ ] Enum conversion
- [ ] Generic types → Rust generics
- [ ] Lambda → closures
- [ ] JSON handling → serde_json
- [ ] Statistics → custom or external crate

---

## Project Structure

```
data_processor.py
├── DataFormat (Enum)
├── DataRecord (Dataclass)
├── DataFilter (Generic class)
├── DataProcessor (Main class)
├── DataValidator (Static methods)
├── DataTransformer (Static methods)
├── DataAnalyzer (Analysis class)
└── Helper functions
```

---

## How to Use

### Run with Python
```bash
python data_processor.py
```

### Expected Output
```
=== Statistics ===
count: 8.00
sum: 361.60
mean: 45.20
median: 43.80
std_dev: 24.13
min: 18.30
max: 75.20

=== Filtering ===
Temperature records: 4
High value records (50-100): 4

=== Grouping ===
sensor_a: 2 records
sensor_b: 2 records
sensor_c: 2 records
sensor_d: 2 records

=== Aggregation ===
sensor_a: 47.60
sensor_b: 37.50
sensor_c: 128.50
sensor_d: 148.00

=== Validation ===
All records valid

=== Transformation ===
Normalized 8 records

=== Analysis ===
Found 2 outliers

Top 3 records:
  sensor_d: 75.20
  sensor_d: 72.80
  sensor_c: 65.00

Percentiles:
  p25: 19.20
  p50: 43.80
  p75: 65.00
  p90: 72.80
  p95: 75.20
  p99: 75.20

=== Export ===
Exported 8 records to JSON (XXX bytes)
```

---

## Translation Steps

### Step 1: Assessment
```bash
# Analyze features and complexity
portalis assess analyze data_processor.py

# Check compatibility
portalis assess compatibility data_processor.py --threshold 0.8
```

### Step 2: Translation
```bash
# Translate to Rust
portalis translate data_processor.py --output data_processor.rs

# Review generated code
cat data_processor.rs
```

### Step 3: Build
```bash
# Build to WASM
portalis build data_processor.rs --target wasm32-wasi

# Or use one-step conversion
portalis convert data_processor.py --output data_processor.wasm
```

### Step 4: Validation
```bash
# Validate WASM
portalis validate data_processor.wasm

# Run and compare output
portalis run data_processor.wasm > wasm_output.txt
python data_processor.py > python_output.txt
diff python_output.txt wasm_output.txt
```

---

## Expected Translation Patterns

### Dataclass → Rust Struct
```python
@dataclass
class DataRecord:
    id: int
    name: str
    value: float
```

Expected Rust:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DataRecord {
    id: i32,
    name: String,
    value: f64,
    timestamp: String,
    tags: Vec<String>,
}
```

### Enum → Rust Enum
```python
class DataFormat(Enum):
    JSON = "json"
    CSV = "csv"
```

Expected Rust:
```rust
#[derive(Debug, Clone)]
enum DataFormat {
    Json,
    Csv,
    Xml,
}
```

### Generic → Rust Generic
```python
class DataFilter(Generic[T]):
    def apply(self, items: List[T]) -> List[T]:
        ...
```

Expected Rust:
```rust
struct DataFilter<T> {
    predicate: Box<dyn Fn(&T) -> bool>,
}

impl<T> DataFilter<T> {
    fn apply(&self, items: Vec<T>) -> Vec<T> {
        items.into_iter()
            .filter(|item| (self.predicate)(item))
            .collect()
    }
}
```

---

## Test Cases

### Test 1: Data Processing Pipeline
```python
processor = DataProcessor()
processor.add_records(create_sample_data())
stats = processor.calculate_statistics()
assert stats['count'] == 8
assert stats['mean'] > 40
```

### Test 2: Filtering
```python
temp = processor.filter_by_tag("temperature")
assert len(temp) == 4

high = processor.filter_by_value_range(50, 100)
assert len(high) == 4
```

### Test 3: JSON Serialization
```python
json_str = processor.to_json()
processor2 = DataProcessor()
processor2.from_json(json_str)
assert len(processor2.records) == 8
```

### Test 4: Data Validation
```python
invalid_record = DataRecord(-1, "", -5.0, "", [])
errors = DataValidator.validate_record(invalid_record)
assert len(errors) >= 3
```

### Test 5: Transformation
```python
records = create_sample_data()
normalized = DataTransformer.normalize_values(records)
assert all(0 <= r.value <= 1 for r in normalized)
```

---

## Performance Expectations

### Python Baseline
- Data loading: ~1ms
- Statistics calculation: ~2ms
- Filtering: ~1ms per filter
- JSON export: ~3ms
- Total: ~10-15ms

### WASM Target
- Data loading: ~0.5ms
- Statistics: ~0.5ms
- Filtering: ~0.3ms per filter
- JSON export: ~1ms
- Total: ~3-5ms
- **Expected Speedup**: 2-3x

### Benchmark
```bash
portalis benchmark data_processor.py --iterations 1000
```

---

## Known Challenges

### High Priority
1. **Generic Types**: May require manual type specification
2. **Lambda Functions**: Should map to closures
3. **JSON Serialization**: Needs serde integration
4. **Statistics Module**: May need custom implementation

### Medium Priority
1. **Dataclass defaults**: May need explicit initialization
2. **Class methods**: @classmethod translation
3. **Static methods**: @staticmethod translation
4. **Type aliases**: TypeVar handling

### Low Priority
1. **String formatting**: f-strings to format! macro
2. **List comprehensions**: Iterator patterns
3. **Dict comprehensions**: HashMap collect patterns

---

## Success Criteria

**Translation Success**:
- [x] All classes translate correctly
- [x] Type system preserved
- [x] Logic flow maintained
- [x] No critical type errors

**Build Success**:
- [x] Rust code compiles
- [x] WASM builds successfully
- [x] No runtime panics

**Functional Correctness**:
- [x] All test cases pass
- [x] Output matches Python
- [x] Edge cases handled
- [x] Statistics accurate to 0.01

**Performance**:
- [x] 2-3x speedup achieved
- [x] Memory usage reasonable
- [x] No performance regressions

---

## Deliverables

After completing this project:
- [x] data_processor.rs (translated Rust code)
- [x] data_processor.wasm (compiled WASM)
- [x] Translation report with metrics
- [x] Test results (all passing)
- [x] Performance comparison
- [x] Feedback on challenges encountered

---

## Beta Testing Focus

### What to Test
1. **Type inference accuracy** for complex types
2. **Generic type handling** (Generic[T])
3. **Dataclass translation** quality
4. **JSON serialization** correctness
5. **Lambda/closure** translation
6. **Statistics library** mapping

### What to Report
1. Any translation failures or warnings
2. Type inference issues
3. Performance deviations from expected
4. Manual adjustments required
5. Documentation gaps

---

## Integration with Beta Program

### Week 3-4: Core Features
- Translate and validate basic functionality
- Test dataclass and enum handling
- Verify JSON serialization

### Week 5-6: Advanced Features
- Test generic types
- Validate statistical operations
- Stress test with larger datasets

### Week 7-8: Production Simulation
- Integrate into CI/CD pipeline
- Performance benchmarking at scale
- Edge case testing

---

## Support

Questions or issues:
- **Slack**: #beta-technical
- **Email**: beta-support@portalis.ai
- **GitHub**: github.com/portalis/beta-issues

Report specific to this project:
- Tag: `[medium-complexity]`
- Include: Translation errors, type issues, performance data
