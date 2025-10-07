//! Comprehensive CPU Bridge Performance Benchmarks
//!
//! This benchmark suite validates CPU Bridge performance against target metrics:
//! - Single file (1KB): < 50ms single core, ~42ms on 16 cores
//! - Small batch (10 files): 500ms → 70ms scaling
//! - Medium batch (100 files): 5s → 500ms scaling
//!
//! Run with: cargo bench --package portalis-cpu-bridge
//! Generate HTML reports: cargo bench --package portalis-cpu-bridge -- --save-baseline main

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use portalis_cpu_bridge::{CpuBridge, CpuConfig, SimdCapability, find_substring, count_char, filter_strings, batch_count_patterns};

// ============================================================================
// Test Data Generation
// ============================================================================

/// Simulates a small Python file translation task (~1KB)
fn create_small_task() -> String {
    // Realistic Python code structure
    let code = r#"
from typing import List, Optional

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Calculator:
    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        return round(a + b, self.precision)

    def multiply(self, a: float, b: float) -> float:
        return round(a * b, self.precision)
"#;
    code.to_string()
}

/// Simulates a medium Python file translation task (~5KB)
fn create_medium_task() -> String {
    let mut code = String::new();
    code.push_str("from typing import List, Dict, Optional\n\n");

    for i in 0..15 {
        code.push_str(&format!(
            r#"
class Processor{i}:
    def __init__(self, config: Dict):
        self.config = config

    def process(self, data: List[Dict]) -> List[Dict]:
        return [self._transform(item) for item in data]

    def _transform(self, item: Dict) -> Dict:
        return {{k: v * 2 for k, v in item.items()}}
"#,
            i = i
        ));
    }

    code
}

/// Simulates a large Python file translation task (~20KB)
fn create_large_task() -> String {
    let mut code = String::new();
    code.push_str("from typing import List, Dict, Optional, Any\nimport asyncio\n\n");

    for i in 0..50 {
        code.push_str(&format!(
            r#"
class Model{i}:
    def __init__(self, id: int, name: str, value: float):
        self.id = id
        self.name = name
        self.value = value

    async def process(self) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {{"id": self.id, "name": self.name, "value": self.value * 2}}
"#,
            i = i
        ));
    }

    code
}

/// Simulate translation work (parsing + type inference + code gen)
fn simulate_translation(source: &str) -> Result<String, anyhow::Error> {
    // Simulate parsing
    let lines: Vec<&str> = source.lines().collect();
    let _line_count = lines.len();

    // Simulate type inference
    let _type_annotations = lines.iter().filter(|l| l.contains("->") || l.contains(":")).count();

    // Simulate code generation
    let output = format!("// Generated Rust code\n// {} lines processed", lines.len());

    Ok(output)
}

// ============================================================================
// Benchmark 1: Single File Translation
// ============================================================================

fn bench_single_file_translation(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_file_translation");

    let bridge = CpuBridge::new();

    // Small file (1KB) - target: <50ms on 4-core
    let small_task = create_small_task();
    group.throughput(Throughput::Bytes(small_task.len() as u64));
    group.bench_function("small_1kb", |b| {
        b.iter(|| {
            let result = bridge.translate_single(black_box(&small_task), |source| {
                simulate_translation(source)
            });
            black_box(result)
        })
    });

    // Medium file (5KB)
    let medium_task = create_medium_task();
    group.throughput(Throughput::Bytes(medium_task.len() as u64));
    group.bench_function("medium_5kb", |b| {
        b.iter(|| {
            let result = bridge.translate_single(black_box(&medium_task), |source| {
                simulate_translation(source)
            });
            black_box(result)
        })
    });

    // Large file (20KB)
    let large_task = create_large_task();
    group.throughput(Throughput::Bytes(large_task.len() as u64));
    group.bench_function("large_20kb", |b| {
        b.iter(|| {
            let result = bridge.translate_single(black_box(&large_task), |source| {
                simulate_translation(source)
            });
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Benchmark 2: Small Batch Processing (10 files)
// ============================================================================

fn bench_small_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_batch_10_files");
    group.throughput(Throughput::Elements(10));

    let bridge = CpuBridge::new();
    let tasks: Vec<String> = (0..10).map(|_| create_small_task()).collect();

    // Sequential baseline
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let results: Vec<_> = tasks
                .iter()
                .map(|t| simulate_translation(t))
                .collect();
            black_box(results)
        })
    });

    // Parallel (target: 500ms → 70ms on 16 cores)
    group.bench_function("parallel", |b| {
        b.iter(|| {
            let result = bridge.parallel_translate(black_box(tasks.clone()), |source| {
                simulate_translation(source)
            });
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Benchmark 3: Medium Batch Processing (100 files)
// ============================================================================

fn bench_medium_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_batch_100_files");
    group.sample_size(20); // Fewer samples for longer benchmarks
    group.throughput(Throughput::Elements(100));

    let bridge = CpuBridge::new();
    let tasks: Vec<String> = (0..100).map(|_| create_small_task()).collect();

    // Sequential baseline
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let results: Vec<_> = tasks
                .iter()
                .map(|t| simulate_translation(t))
                .collect();
            black_box(results)
        })
    });

    // Parallel (target: 5s → 500ms on 16 cores)
    group.bench_function("parallel", |b| {
        b.iter(|| {
            let result = bridge.parallel_translate(black_box(tasks.clone()), |source| {
                simulate_translation(source)
            });
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Benchmark 4: Thread Scaling (1, 2, 4, 8, 16 cores)
// ============================================================================

fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");
    group.sample_size(20);
    group.throughput(Throughput::Elements(100));

    let tasks: Vec<String> = (0..100).map(|_| create_small_task()).collect();

    // Test with different thread counts
    for num_threads in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            num_threads,
            |b, &num_threads| {
                let config = CpuConfig::builder().num_threads(num_threads).build();
                let bridge = CpuBridge::with_config(config);

                b.iter(|| {
                    let result = bridge.parallel_translate(black_box(tasks.clone()), |source| {
                        simulate_translation(source)
                    });
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 5: Workload Complexity Scaling
// ============================================================================

fn bench_workload_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload_complexity");

    let bridge = CpuBridge::new();

    let test_cases = vec![
        ("small_1kb", create_small_task()),
        ("medium_5kb", create_medium_task()),
        ("large_20kb", create_large_task()),
    ];

    for (name, source) in test_cases {
        group.throughput(Throughput::Bytes(source.len() as u64));

        // Single file
        group.bench_with_input(BenchmarkId::new("single", name), &source, |b, src| {
            b.iter(|| {
                let result = bridge.translate_single(black_box(src), |s| simulate_translation(s));
                black_box(result)
            });
        });

        // Batch of 10
        let batch: Vec<String> = (0..10).map(|_| source.clone()).collect();
        group.bench_with_input(BenchmarkId::new("batch_10", name), &batch, |b, tasks| {
            b.iter(|| {
                let result = bridge.parallel_translate(black_box(tasks.clone()), |s| {
                    simulate_translation(s)
                });
                black_box(result)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Benchmark 6: Memory Efficiency
// ============================================================================

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    let bridge = CpuBridge::new();

    for batch_size in [10, 50, 100, 500].iter() {
        let tasks: Vec<String> = (0..*batch_size).map(|_| create_small_task()).collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &tasks,
            |b, tasks| {
                b.iter(|| {
                    let result = bridge.parallel_translate(black_box(tasks.clone()), |s| {
                        simulate_translation(s)
                    });
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 7: Realistic Mixed Workload
// ============================================================================

fn bench_realistic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_workload");
    group.sample_size(10);
    group.throughput(Throughput::Elements(100));

    let bridge = CpuBridge::new();

    // Mix of file sizes (realistic project)
    let mut tasks = Vec::new();

    // 70% small files
    for _ in 0..70 {
        tasks.push(create_small_task());
    }

    // 25% medium files
    for _ in 70..95 {
        tasks.push(create_medium_task());
    }

    // 5% large files
    for _ in 95..100 {
        tasks.push(create_large_task());
    }

    group.bench_function("mixed_100_files", |b| {
        b.iter(|| {
            let result = bridge.parallel_translate(black_box(tasks.clone()), |s| {
                simulate_translation(s)
            });
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Benchmark 8: CPU Bridge Overhead
// ============================================================================

fn bench_cpu_bridge_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_bridge_overhead");

    let bridge = CpuBridge::new();
    let task = create_small_task();

    // Raw translation (baseline)
    group.bench_function("raw_translation", |b| {
        b.iter(|| black_box(simulate_translation(&task)))
    });

    // With CPU bridge (measure overhead, target: <10%)
    group.bench_function("with_bridge", |b| {
        b.iter(|| {
            let result = bridge.translate_single(black_box(&task), |s| simulate_translation(s));
            black_box(result)
        })
    });

    // Batch overhead
    let tasks: Vec<String> = (0..10).map(|_| task.clone()).collect();
    group.throughput(Throughput::Elements(10));

    group.bench_function("batch_overhead", |b| {
        b.iter(|| {
            let result = bridge.parallel_translate(black_box(tasks.clone()), |s| {
                simulate_translation(s)
            });
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Benchmark 9: Simple Integer Operations (existing)
// ============================================================================

fn bench_simple_operations(c: &mut Criterion) {
    let bridge = CpuBridge::new();

    c.bench_function("simple_single_task", |b| {
        b.iter(|| {
            let result = bridge.translate_single(black_box(42), |&x| Ok(x * 2));
            black_box(result)
        })
    });

    let mut group = c.benchmark_group("simple_parallel_tasks");

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let tasks: Vec<i32> = (0..size).collect();
            b.iter(|| {
                let result = bridge.parallel_translate(black_box(tasks.clone()), |&x| Ok(x * 2));
                black_box(result)
            });
        });
    }
    group.finish();
}

// ============================================================================
// SIMD Benchmark 1: String Matching (SIMD vs Scalar)
// ============================================================================

fn bench_simd_string_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_string_matching");

    // Detect SIMD capabilities
    let capability = SimdCapability::detect();
    println!("\n=== SIMD Capability: {} ({}B vector width) ===",
             capability.name(), capability.vector_width());

    // Test cases with different lengths
    let test_cases = vec![
        ("short_import", "import numpy as np\nimport pandas as pd", "numpy"),
        ("medium_code",
         &"from typing import List, Dict, Optional\n".repeat(20),
         "Optional"),
        ("long_file",
         &"def function_name(arg1: int, arg2: str) -> bool:\n    return True\n".repeat(100),
         "return"),
    ];

    for (name, text, pattern) in test_cases {
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("simd", name),
            &(text, pattern),
            |b, (text, pattern)| {
                b.iter(|| {
                    let result = find_substring(black_box(text), black_box(pattern));
                    black_box(result)
                });
            },
        );

        // Scalar baseline for comparison
        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &(text, pattern),
            |b, (text, pattern)| {
                b.iter(|| {
                    let result = text.find(black_box(pattern));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SIMD Benchmark 2: Batch String Operations
// ============================================================================

fn bench_simd_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_batch_operations");

    // Generate realistic import statements
    let imports = vec![
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from typing import List, Dict",
        "import torch",
        "import tensorflow as tf",
        "from collections import defaultdict",
        "import asyncio",
    ];

    // Test with different batch sizes
    for batch_size in [10, 100, 1000] {
        let strings: Vec<String> = (0..batch_size)
            .map(|i| imports[i % imports.len()].to_string())
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        // SIMD filter
        group.bench_with_input(
            BenchmarkId::new("simd_filter", batch_size),
            &strings,
            |b, strings| {
                b.iter(|| {
                    let result = filter_strings(black_box(strings), black_box("import"));
                    black_box(result)
                });
            },
        );

        // Scalar filter baseline
        group.bench_with_input(
            BenchmarkId::new("scalar_filter", batch_size),
            &strings,
            |b, strings| {
                b.iter(|| {
                    let result: Vec<&String> = strings
                        .iter()
                        .filter(|s| s.contains(black_box("import")))
                        .collect();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SIMD Benchmark 3: Character Counting
// ============================================================================

fn bench_simd_char_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_char_counting");

    let test_cases = vec![
        ("small_file", create_small_task(), ':'),
        ("medium_file", create_medium_task(), ':'),
        ("large_file", create_large_task(), ':'),
    ];

    for (name, text, target_char) in test_cases {
        group.throughput(Throughput::Bytes(text.len() as u64));

        // SIMD count
        group.bench_with_input(
            BenchmarkId::new("simd", name),
            &(text.clone(), target_char),
            |b, (text, target_char)| {
                b.iter(|| {
                    let count = count_char(black_box(text), black_box(*target_char));
                    black_box(count)
                });
            },
        );

        // Scalar count baseline
        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &(text, target_char),
            |b, (text, target_char)| {
                b.iter(|| {
                    let count = text.chars().filter(|&c| c == black_box(*target_char)).count();
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SIMD Benchmark 4: Import Statement Matching (100K imports)
// ============================================================================

fn bench_simd_import_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_import_matching");
    group.sample_size(20);

    // Generate 100K import statements
    let import_templates = vec![
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import tensorflow as tf",
        "import torch",
        "import scipy",
        "import sklearn",
        "from typing import List",
    ];

    let imports: Vec<String> = (0..100_000)
        .map(|i| import_templates[i % import_templates.len()].to_string())
        .collect();

    group.throughput(Throughput::Elements(100_000));

    // Find all imports containing "numpy"
    group.bench_function("find_numpy_100k", |b| {
        b.iter(|| {
            let result = filter_strings(black_box(&imports), black_box("numpy"));
            black_box(result)
        });
    });

    // Find all imports containing "typing"
    group.bench_function("find_typing_100k", |b| {
        b.iter(|| {
            let result = filter_strings(black_box(&imports), black_box("typing"));
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// SIMD Benchmark 5: Identifier Filtering (50K identifiers)
// ============================================================================

fn bench_simd_identifier_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_identifier_filtering");
    group.sample_size(20);

    // Generate 50K Python identifiers
    let identifier_templates = vec![
        "my_function", "calculate_value", "process_data", "validate_input",
        "parse_config", "serialize_json", "deserialize_object", "transform_array",
        "filter_results", "map_values", "reduce_collection", "aggregate_stats",
    ];

    let identifiers: Vec<String> = (0..50_000)
        .map(|i| format!("{}_{}", identifier_templates[i % identifier_templates.len()], i))
        .collect();

    group.throughput(Throughput::Elements(50_000));

    // Filter identifiers containing "process"
    group.bench_function("filter_process_50k", |b| {
        b.iter(|| {
            let result = filter_strings(black_box(&identifiers), black_box("process"));
            black_box(result)
        });
    });

    // Filter identifiers containing "data"
    group.bench_function("filter_data_50k", |b| {
        b.iter(|| {
            let result = filter_strings(black_box(&identifiers), black_box("data"));
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// SIMD Benchmark 6: Type Annotation Parsing (10K type strings)
// ============================================================================

fn bench_simd_type_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_type_parsing");
    group.sample_size(20);

    // Generate 10K type annotation strings
    let type_templates = vec![
        "List[int]",
        "Dict[str, Any]",
        "Optional[str]",
        "Union[int, float]",
        "Tuple[str, int, bool]",
        "Callable[[int, str], bool]",
        "List[Dict[str, Any]]",
        "Optional[List[str]]",
    ];

    let type_annotations: Vec<String> = (0..10_000)
        .map(|i| type_templates[i % type_templates.len()].to_string())
        .collect();

    group.throughput(Throughput::Elements(10_000));

    // Find all Optional types
    group.bench_function("find_optional_10k", |b| {
        b.iter(|| {
            let result = filter_strings(black_box(&type_annotations), black_box("Optional"));
            black_box(result)
        });
    });

    // Find all List types
    group.bench_function("find_list_10k", |b| {
        b.iter(|| {
            let result = filter_strings(black_box(&type_annotations), black_box("List"));
            black_box(result)
        });
    });

    // Count bracket occurrences
    group.bench_function("count_brackets_10k", |b| {
        b.iter(|| {
            let mut total = 0;
            for type_str in &type_annotations {
                total += count_char(black_box(type_str), black_box('['));
            }
            black_box(total)
        });
    });

    group.finish();
}

// ============================================================================
// SIMD Benchmark 7: Pattern Occurrence Counting
// ============================================================================

fn bench_simd_pattern_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_pattern_counting");
    group.sample_size(20);

    // Create strings with multiple occurrences of patterns
    let test_strings: Vec<String> = vec![
        "def func1(): def func2(): def func3():".repeat(100),
        "class A: class B: class C:".repeat(100),
        "import x; import y; import z;".repeat(100),
    ];

    group.throughput(Throughput::Elements(test_strings.len() as u64));

    // Count "def" occurrences
    group.bench_function("count_def_pattern", |b| {
        b.iter(|| {
            let counts = batch_count_patterns(black_box(&test_strings), black_box("def"));
            black_box(counts)
        });
    });

    // Count "class" occurrences
    group.bench_function("count_class_pattern", |b| {
        b.iter(|| {
            let counts = batch_count_patterns(black_box(&test_strings), black_box("class"));
            black_box(counts)
        });
    });

    group.finish();
}

// ============================================================================
// SIMD Benchmark 8: AST Node Filtering Simulation
// ============================================================================

fn bench_simd_ast_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_ast_filtering");

    // Simulate AST node representations as strings
    let ast_nodes: Vec<String> = vec![
        "FunctionDef(name='process_data', args=['x', 'y'])".to_string(),
        "ClassDef(name='DataProcessor', bases=['BaseProcessor'])".to_string(),
        "Import(module='numpy', alias='np')".to_string(),
        "Assign(target='result', value='42')".to_string(),
        "Call(func='print', args=['hello'])".to_string(),
    ];

    // Create large dataset
    let large_ast: Vec<String> = (0..10_000)
        .map(|i| ast_nodes[i % ast_nodes.len()].clone())
        .collect();

    group.throughput(Throughput::Elements(large_ast.len() as u64));

    // Filter function definitions
    group.bench_function("filter_function_defs", |b| {
        b.iter(|| {
            let result = filter_strings(black_box(&large_ast), black_box("FunctionDef"));
            black_box(result)
        });
    });

    // Filter imports
    group.bench_function("filter_imports", |b| {
        b.iter(|| {
            let result = filter_strings(black_box(&large_ast), black_box("Import"));
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// SIMD Benchmark 9: Platform-Specific Performance Matrix
// ============================================================================

fn bench_simd_platform_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_platform_performance");

    let capability = SimdCapability::detect();

    // Create test data sized to vector width
    let vector_width = capability.vector_width();
    let test_sizes = vec![
        vector_width,
        vector_width * 2,
        vector_width * 4,
        vector_width * 8,
        vector_width * 16,
    ];

    for size in test_sizes {
        let text = "x".repeat(size);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("aligned_search", size),
            &text,
            |b, text| {
                b.iter(|| {
                    let result = find_substring(black_box(text), black_box("x"));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SIMD Benchmark 10: Throughput Comparison (ops/sec)
// ============================================================================

fn bench_simd_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_throughput");
    group.sample_size(50);

    // Large text corpus for throughput testing
    let corpus = create_large_task().repeat(10); // ~200KB

    group.throughput(Throughput::Bytes(corpus.len() as u64));

    // String search throughput
    group.bench_function("string_search_throughput", |b| {
        b.iter(|| {
            let result = find_substring(black_box(&corpus), black_box("async"));
            black_box(result)
        });
    });

    // Character count throughput
    group.bench_function("char_count_throughput", |b| {
        b.iter(|| {
            let count = count_char(black_box(&corpus), black_box(':'));
            black_box(count)
        });
    });

    group.finish();
}

// ============================================================================
// SIMD Benchmark 11: Latency Comparison (time/operation)
// ============================================================================

fn bench_simd_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_latency");
    group.sample_size(100);

    // Small operations for latency testing
    let small_text = "import numpy as np";

    group.bench_function("single_search_latency", |b| {
        b.iter(|| {
            let result = find_substring(black_box(small_text), black_box("numpy"));
            black_box(result)
        });
    });

    group.bench_function("single_count_latency", |b| {
        b.iter(|| {
            let count = count_char(black_box(small_text), black_box('p'));
            black_box(count)
        });
    });

    group.finish();
}

// ============================================================================
// SIMD Benchmark 12: Real-world Mixed Workload
// ============================================================================

fn bench_simd_realistic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_realistic_workload");
    group.sample_size(20);

    // Mix of operations that would occur during transpilation
    let python_files: Vec<String> = (0..100)
        .map(|_| create_medium_task())
        .collect();

    group.throughput(Throughput::Elements(python_files.len() as u64));

    group.bench_function("transpilation_simulation", |b| {
        b.iter(|| {
            // 1. Filter files with specific imports
            let has_typing = filter_strings(black_box(&python_files), "typing");

            // 2. Count type annotations
            let mut annotation_count = 0;
            for file in &python_files {
                annotation_count += count_char(file, ':');
            }

            // 3. Find all class definitions
            let classes = filter_strings(black_box(&python_files), "class");

            black_box((has_typing, annotation_count, classes))
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_single_file_translation,
    bench_small_batch,
    bench_medium_batch,
    bench_thread_scaling,
    bench_workload_complexity,
    bench_memory_efficiency,
    bench_realistic_workload,
    bench_cpu_bridge_overhead,
    bench_simple_operations,
);

criterion_group!(
    simd_benches,
    bench_simd_string_matching,
    bench_simd_batch_operations,
    bench_simd_char_counting,
    bench_simd_import_matching,
    bench_simd_identifier_filtering,
    bench_simd_type_parsing,
    bench_simd_pattern_counting,
    bench_simd_ast_filtering,
    bench_simd_platform_performance,
    bench_simd_throughput,
    bench_simd_latency,
    bench_simd_realistic_workload,
);

criterion_main!(benches, simd_benches);
