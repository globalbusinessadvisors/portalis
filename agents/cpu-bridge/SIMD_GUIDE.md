# SIMD Optimization Guide

## Overview

The CPU Bridge SIMD module provides comprehensive Single Instruction Multiple Data (SIMD) optimizations for CPU-based parallel processing. It leverages platform-specific instruction sets to accelerate string operations, pattern matching, and character counting tasks.

## Architecture

### Supported Platforms

| Platform | Instruction Set | Vector Width | Speedup vs Scalar |
|----------|----------------|--------------|-------------------|
| x86_64   | AVX2           | 256-bit      | 2-4x              |
| x86_64   | SSE4.2         | 128-bit      | 1.5-2.5x          |
| ARM64    | NEON           | 128-bit      | 2-3x              |
| Other    | Scalar (fallback) | -         | 1x (baseline)     |

### Runtime Detection

The module automatically detects available CPU features at runtime and selects the best implementation:

```rust
use portalis_cpu_bridge::simd::detect_cpu_capabilities;

let caps = detect_cpu_capabilities();
println!("Best SIMD: {}", caps.best_simd());
println!("AVX2 available: {}", caps.avx2);
println!("SSE4.2 available: {}", caps.sse42);
println!("NEON available: {}", caps.neon);
```

**Detection Overhead:** < 1μs (cached after first call)

## API Reference

### `batch_string_contains`

Checks if each string in a haystack contains a needle substring using SIMD acceleration.

```rust
use portalis_cpu_bridge::simd::batch_string_contains;

let haystack = vec![
    "import std::io",
    "use rayon",
    "import numpy",
    "fn main()"
];

let results = batch_string_contains(&haystack, "import");
// Results: [true, false, true, false]
```

**Performance Characteristics:**
- **AVX2**: 3-4x faster than scalar for large batches
- **NEON**: 2-3x faster than scalar
- **Best for**: Batches of 10+ strings

### `parallel_string_match`

Performs parallel string prefix matching using SIMD acceleration.

```rust
use portalis_cpu_bridge::simd::parallel_string_match;

let strings = vec!["test_123", "test_456", "example", "test"];
let matches = parallel_string_match(&strings, "test_");
// Results: [true, true, false, false]
```

**Performance Characteristics:**
- **AVX2**: 3-4x faster than scalar for patterns up to 32 bytes
- **SSE4.2**: 2-3x faster for patterns up to 16 bytes
- **NEON**: 2-3x faster for patterns up to 16 bytes
- **Optimized for**: Short patterns (< 32 chars)

**Pattern Length Impact:**
- 1-16 bytes: Maximum SIMD benefit (all platforms)
- 17-32 bytes: AVX2 optimization (x86_64 only)
- 33+ bytes: Automatic scalar fallback

### `vectorized_char_count`

Counts occurrences of a character in multiple strings using SIMD.

```rust
use portalis_cpu_bridge::simd::vectorized_char_count;

let strings = vec!["hello world", "test string", "aaa"];
let counts = vectorized_char_count(&strings, 'l');
// Results: [3, 0, 0]
```

**Performance Characteristics:**
- **AVX2**: 4x faster than scalar for ASCII characters
- **NEON**: 3x faster than scalar
- **Best for**: Large strings (> 64 bytes), ASCII characters
- **Unicode**: Automatically uses scalar fallback

## Performance Optimization Tips

### 1. Batch Processing

Process strings in batches for maximum SIMD efficiency:

```rust
// Good: Batch processing
let haystack: Vec<&str> = vec![/* many strings */];
let results = batch_string_contains(&haystack, "pattern");

// Less efficient: Individual processing
for s in haystack {
    let result = s.contains("pattern"); // No SIMD
}
```

### 2. Pattern Length

Keep patterns short for optimal SIMD performance:

```rust
// Optimal: Short patterns leverage SIMD fully
parallel_string_match(&strings, "test_");  // 5 bytes - full SIMD

// Less optimal: Long patterns use partial SIMD or scalar
parallel_string_match(&strings, "very_long_pattern_string");  // 25+ bytes
```

### 3. ASCII vs Unicode

SIMD optimizations work best with ASCII:

```rust
// Optimal: ASCII character counting uses SIMD
vectorized_char_count(&strings, 'a');  // Fast

// Fallback: Unicode uses scalar implementation
vectorized_char_count(&strings, '世');  // Slower (but still correct)
```

### 4. String Length

SIMD benefits increase with string length:

| String Length | SIMD Benefit |
|--------------|--------------|
| < 16 bytes   | Minimal      |
| 16-64 bytes  | Moderate     |
| 64+ bytes    | Maximum      |

## Platform-Specific Details

### x86_64 (AVX2)

**Capabilities:**
- 256-bit vector operations
- Process 32 bytes simultaneously
- Best overall performance

**Detection:**
```rust
#[cfg(target_arch = "x86_64")]
{
    if is_x86_feature_detected!("avx2") {
        println!("AVX2 available!");
    }
}
```

### x86_64 (SSE4.2)

**Capabilities:**
- 128-bit vector operations
- Process 16 bytes simultaneously
- Specialized string comparison instructions

**Use Case:** Older x86_64 CPUs without AVX2

### ARM64 (NEON)

**Capabilities:**
- 128-bit vector operations (mandatory on AArch64)
- Process 16 bytes simultaneously
- Optimized for mobile and embedded systems

**Detection:**
```rust
#[cfg(target_arch = "aarch64")]
{
    // NEON is always available on ARM64
    println!("NEON available!");
}
```

## Integration Examples

### AST Processing

```rust
use portalis_cpu_bridge::simd::parallel_string_match;

// Filter import statements
let lines = vec![
    "import std::io",
    "use rayon",
    "import numpy as np",
    "fn main() {}"
];

let imports = parallel_string_match(&lines, "import");
let import_lines: Vec<&str> = lines.iter()
    .zip(imports.iter())
    .filter_map(|(line, &is_import)| if is_import { Some(*line) } else { None })
    .collect();
```

### Identifier Analysis

```rust
use portalis_cpu_bridge::simd::{batch_string_contains, vectorized_char_count};

// Find identifiers containing specific patterns
let identifiers = vec!["user_id", "user_name", "admin_role", "get_user"];
let user_related = batch_string_contains(&identifiers, "user");

// Count naming convention usage (snake_case underscores)
let underscore_counts = vectorized_char_count(&identifiers, '_');
```

### Code Search

```rust
use portalis_cpu_bridge::simd::batch_string_contains;

// Search for TODO comments in code
let code_lines = vec![
    "// TODO: Implement feature",
    "fn process() {",
    "    // FIXME: Handle edge case",
    "    // TODO: Add validation"
];

let todos = batch_string_contains(&code_lines, "TODO");
let todo_lines: Vec<&str> = code_lines.iter()
    .zip(todos.iter())
    .filter_map(|(line, &has_todo)| if has_todo { Some(*line) } else { None })
    .collect();
```

## Safety Guarantees

### Zero Unsafe Public API

All SIMD operations are wrapped in safe Rust APIs. Unsafe code is:
- Encapsulated in private functions
- Protected with `#[target_feature]` attributes
- Runtime-dispatched based on CPU capabilities
- Thoroughly tested with both SIMD and scalar paths

### Fallback Guarantees

- **Always functional**: Scalar fallback ensures operations work on all CPUs
- **No panic**: Invalid inputs use standard Rust error handling
- **Memory safe**: No buffer overflows or out-of-bounds access
- **Thread safe**: All operations are thread-safe and reentrant

## Benchmarking

Run comprehensive benchmarks:

```bash
# Run all SIMD benchmarks
cargo bench --bench simd_benchmarks

# Generate HTML reports
cargo bench --bench simd_benchmarks -- --save-baseline my-baseline

# Compare against baseline
cargo bench --bench simd_benchmarks -- --baseline my-baseline
```

### Expected Results (x86_64 AVX2)

```
batch_string_contains/10     time: [1.2 μs]   throughput: 8.3 Melem/s
batch_string_contains/100    time: [8.5 μs]   throughput: 11.8 Melem/s
batch_string_contains/1000   time: [78 μs]    throughput: 12.8 Melem/s

parallel_string_match/10     time: [950 ns]   throughput: 10.5 Melem/s
parallel_string_match/100    time: [7.2 μs]   throughput: 13.9 Melem/s
parallel_string_match/1000   time: [68 μs]    throughput: 14.7 Melem/s

vectorized_char_count/10     time: [1.5 μs]   throughput: 6.7 Melem/s
vectorized_char_count/100    time: [12 μs]    throughput: 8.3 Melem/s
vectorized_char_count/1000   time: [115 μs]   throughput: 8.7 Melem/s
```

## Compatibility Matrix

| Platform      | AVX2 | SSE4.2 | NEON | Scalar |
|---------------|------|--------|------|--------|
| x86_64 Modern | ✅   | ✅     | ❌   | ✅     |
| x86_64 Legacy | ❌   | ✅     | ❌   | ✅     |
| ARM64         | ❌   | ❌     | ✅   | ✅     |
| Other         | ❌   | ❌     | ❌   | ✅     |

**Legend:**
- ✅ Supported
- ❌ Not supported (automatic fallback)

## Future Enhancements

Planned optimizations for future releases:

1. **AVX-512 Support**: 512-bit vectors for next-gen Intel CPUs
2. **SVE Support**: Scalable Vector Extension for ARM
3. **Advanced Pattern Matching**: Boyer-Moore with SIMD
4. **Parallel Parsing**: SIMD-accelerated JSON/XML parsing
5. **Compression**: SIMD-accelerated string compression

## Troubleshooting

### SIMD Not Detected

If SIMD features aren't detected when they should be:

```rust
let caps = detect_cpu_capabilities();
if !caps.has_simd() {
    eprintln!("Warning: No SIMD support detected");
    eprintln!("Platform: {}", std::env::consts::ARCH);
    eprintln!("Using scalar fallback");
}
```

### Performance Not Improving

Common issues:
1. **Small batches**: SIMD needs ≥10 items for overhead amortization
2. **Short strings**: Strings < 16 bytes see minimal benefit
3. **Unicode text**: Non-ASCII disables some optimizations
4. **Debug build**: Always benchmark in release mode

### Build Issues

Ensure you're using a recent Rust version:

```bash
rustc --version  # Should be 1.70+
cargo build --release
```

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Rust std::arch Documentation](https://doc.rust-lang.org/std/arch/)
