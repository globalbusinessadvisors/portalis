use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use portalis_cuda_bridge::{CudaParser, BatchParser, CudaParserConfig};

/// Small Python code sample (~100 lines)
const SMALL_PYTHON: &str = r#"
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Calculator:
    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        return round(a + b, self.precision)

    def subtract(self, a: float, b: float) -> float:
        return round(a - b, self.precision)

    def multiply(self, a: float, b: float) -> float:
        return round(a * b, self.precision)

def main():
    calc = Calculator()
    result = calc.add(10.5, 20.3)
    print(f"Result: {result}")
"#;

/// Medium Python code sample (~500 lines)
fn generate_medium_python() -> String {
    let mut code = String::new();
    code.push_str("from typing import List, Dict, Optional\n\n");

    for i in 0..25 {
        code.push_str(&format!(r#"
class DataProcessor{i}:
    """Data processor class {i}"""

    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.results = []

    def process(self, data: List[Dict]) -> List[Dict]:
        """Process input data."""
        processed = []
        for item in data:
            result = self._transform(item)
            if self._validate(result):
                processed.append(result)
        return processed

    def _transform(self, item: Dict) -> Dict:
        """Transform single item."""
        return {{k: v * 2 if isinstance(v, (int, float)) else v
                 for k, v in item.items()}}

    def _validate(self, item: Dict) -> bool:
        """Validate transformed item."""
        return all(v is not None for v in item.values())
"#, i = i));
    }

    code
}

/// Large Python code sample (~2000 lines)
fn generate_large_python() -> String {
    let mut code = String::new();
    code.push_str("from typing import List, Dict, Optional, Tuple, Set\n");
    code.push_str("import asyncio\nimport dataclasses\n\n");

    for i in 0..100 {
        code.push_str(&format!(r#"
@dataclasses.dataclass
class Model{i}:
    """Data model {i}"""
    id: int
    name: str
    value: float
    tags: List[str]
    metadata: Dict[str, any]

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Model{i}':
        return cls(**data)

class Service{i}:
    """Service class {i}"""

    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.cache = {{}}
        self.stats = {{'processed': 0, 'errors': 0}}

    async def process_batch(self, items: List[Model{i}]) -> List[Dict]:
        """Process batch of items asynchronously."""
        results = []
        for item in items:
            try:
                result = await self._process_single(item)
                results.append(result)
                self.stats['processed'] += 1
            except Exception as e:
                self.stats['errors'] += 1
                print(f"Error processing {{item.id}}: {{e}}")
        return results

    async def _process_single(self, item: Model{i}) -> Dict:
        """Process single item."""
        await asyncio.sleep(0)  # Yield control
        return item.to_dict()
"#, i = i));
    }

    code
}

/// Benchmark AST parsing for different file sizes
fn bench_parsing_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing_by_size");

    let parser = CudaParser::new().unwrap();

    // Small file
    group.throughput(Throughput::Bytes(SMALL_PYTHON.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("small", SMALL_PYTHON.len()),
        &SMALL_PYTHON,
        |b, source| {
            b.iter(|| {
                parser.parse(black_box(source)).unwrap()
            });
        },
    );

    // Medium file
    let medium_code = generate_medium_python();
    group.throughput(Throughput::Bytes(medium_code.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("medium", medium_code.len()),
        &medium_code,
        |b, source| {
            b.iter(|| {
                parser.parse(black_box(source)).unwrap()
            });
        },
    );

    // Large file
    let large_code = generate_large_python();
    group.throughput(Throughput::Bytes(large_code.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("large", large_code.len()),
        &large_code,
        |b, source| {
            b.iter(|| {
                parser.parse(black_box(source)).unwrap()
            });
        },
    );

    group.finish();
}

/// Benchmark batch processing
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    let batch_parser = BatchParser::new().unwrap();

    // Small batches
    for batch_size in [10, 50, 100].iter() {
        let sources: Vec<&str> = (0..*batch_size)
            .map(|_| SMALL_PYTHON)
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &sources,
            |b, sources| {
                b.iter(|| {
                    batch_parser.parse_batch(black_box(sources)).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different parser configurations
fn bench_parser_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser_configs");

    let configs = vec![
        ("default", CudaParserConfig::default()),
        ("high_capacity", CudaParserConfig {
            max_nodes: 500_000,
            max_tokens: 2_000_000,
            max_depth: 5_000,
            collect_metrics: true,
        }),
        ("low_overhead", CudaParserConfig {
            max_nodes: 10_000,
            max_tokens: 50_000,
            max_depth: 100,
            collect_metrics: false,
        }),
    ];

    let medium_code = generate_medium_python();

    for (name, config) in configs {
        let parser = CudaParser::with_config(config).unwrap();

        group.bench_with_input(
            BenchmarkId::new("config", name),
            &medium_code,
            |b, source| {
                b.iter(|| {
                    parser.parse(black_box(source)).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    let parser = CudaParser::new().unwrap();

    // Test memory stats retrieval overhead
    group.bench_function("memory_stats_retrieval", |b| {
        b.iter(|| {
            black_box(parser.memory_stats())
        });
    });

    // Test parsing with metrics collection
    let code = generate_medium_python();
    group.bench_function("parse_with_metrics", |b| {
        b.iter(|| {
            let result = parser.parse(black_box(&code)).unwrap();
            black_box(result.metrics)
        });
    });

    group.finish();
}

/// Benchmark CPU vs GPU simulation
fn bench_cpu_vs_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_vs_gpu_comparison");

    // CPU-only parser (no GPU available in this environment)
    let cpu_parser = CudaParser::new().unwrap();

    let test_code = generate_medium_python();

    group.bench_function("cpu_fallback", |b| {
        b.iter(|| {
            cpu_parser.parse(black_box(&test_code)).unwrap()
        });
    });

    // Note: GPU benchmarks would go here if CUDA feature was enabled
    // For now, we document expected GPU performance based on research

    group.finish();
}

/// Benchmark tokenization vs AST construction
fn bench_parsing_phases(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing_phases");

    let parser = CudaParser::new().unwrap();
    let code = generate_medium_python();

    group.bench_function("full_parse", |b| {
        b.iter(|| {
            let result = parser.parse(black_box(&code)).unwrap();
            black_box(result)
        });
    });

    group.bench_function("parse_and_extract_metrics", |b| {
        b.iter(|| {
            let result = parser.parse(black_box(&code)).unwrap();
            (
                black_box(result.metrics.tokenization_time_ms),
                black_box(result.metrics.parsing_time_ms),
                black_box(result.metrics.nodes_created),
            )
        });
    });

    group.finish();
}

/// Benchmark scalability with increasing complexity
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");

    let parser = CudaParser::new().unwrap();

    // Test with increasing number of classes
    for num_classes in [10, 50, 100, 200].iter() {
        let mut code = String::new();
        code.push_str("from typing import List, Dict\n\n");

        for i in 0..*num_classes {
            code.push_str(&format!(r#"
class Class{i}:
    def method{i}(self, x: int) -> int:
        return x * {i}
"#, i = i));
        }

        group.throughput(Throughput::Elements(*num_classes as u64));
        group.bench_with_input(
            BenchmarkId::new("num_classes", num_classes),
            &code,
            |b, source| {
                b.iter(|| {
                    parser.parse(black_box(source)).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parsing_by_size,
    bench_batch_processing,
    bench_parser_configs,
    bench_memory_usage,
    bench_cpu_vs_gpu,
    bench_parsing_phases,
    bench_scalability,
);

criterion_main!(benches);
