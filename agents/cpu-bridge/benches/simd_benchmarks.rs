use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use portalis_cpu_bridge::simd::{
    batch_string_contains, detect_cpu_capabilities, parallel_string_match, vectorized_char_count,
};

fn benchmark_batch_string_contains(c: &mut Criterion) {
    let caps = detect_cpu_capabilities();
    let mut group = c.benchmark_group("batch_string_contains");

    // Create test data of various sizes
    let sizes = [10, 100, 1000];

    for size in sizes.iter() {
        let haystack: Vec<String> = (0..*size)
            .map(|i| format!("import module_{} from 'package'", i))
            .collect();
        let haystack_refs: Vec<&str> = haystack.iter().map(|s| s.as_str()).collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new(caps.best_simd(), size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(batch_string_contains(
                        black_box(&haystack_refs),
                        black_box("import"),
                    ))
                });
            },
        );
    }
    group.finish();
}

fn benchmark_parallel_string_match(c: &mut Criterion) {
    let caps = detect_cpu_capabilities();
    let mut group = c.benchmark_group("parallel_string_match");

    let sizes = [10, 100, 1000];

    for size in sizes.iter() {
        let strings: Vec<String> = (0..*size)
            .map(|i| format!("test_function_{}", i))
            .collect();
        let string_refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new(caps.best_simd(), size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(parallel_string_match(
                        black_box(&string_refs),
                        black_box("test_"),
                    ))
                });
            },
        );
    }
    group.finish();
}

fn benchmark_vectorized_char_count(c: &mut Criterion) {
    let caps = detect_cpu_capabilities();
    let mut group = c.benchmark_group("vectorized_char_count");

    let sizes = [10, 100, 1000];

    for size in sizes.iter() {
        let strings: Vec<String> = (0..*size)
            .map(|i| "The quick brown fox jumps over the lazy dog".repeat(i % 5 + 1))
            .collect();
        let string_refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new(caps.best_simd(), size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(vectorized_char_count(
                        black_box(&string_refs),
                        black_box('o'),
                    ))
                });
            },
        );
    }
    group.finish();
}

fn benchmark_short_vs_long_patterns(c: &mut Criterion) {
    let caps = detect_cpu_capabilities();
    let mut group = c.benchmark_group("pattern_length_comparison");

    let strings: Vec<String> = (0..100)
        .map(|i| format!("this_is_a_very_long_identifier_name_for_testing_{}", i))
        .collect();
    let string_refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

    group.bench_function(
        BenchmarkId::new(format!("{}_short_pattern", caps.best_simd()), "4chars"),
        |b| {
            b.iter(|| {
                black_box(parallel_string_match(
                    black_box(&string_refs),
                    black_box("this"),
                ))
            });
        },
    );

    group.bench_function(
        BenchmarkId::new(format!("{}_long_pattern", caps.best_simd()), "32chars"),
        |b| {
            b.iter(|| {
                black_box(parallel_string_match(
                    black_box(&string_refs),
                    black_box("this_is_a_very_long_identifier"),
                ))
            });
        },
    );

    group.finish();
}

fn benchmark_ascii_vs_unicode(c: &mut Criterion) {
    let mut group = c.benchmark_group("ascii_vs_unicode");

    let ascii_strings: Vec<String> = (0..100).map(|i| format!("test string {}", i)).collect();
    let ascii_refs: Vec<&str> = ascii_strings.iter().map(|s| s.as_str()).collect();

    let unicode_strings: Vec<String> = (0..100)
        .map(|i| format!("测试字符串 {} 🚀", i))
        .collect();
    let unicode_refs: Vec<&str> = unicode_strings.iter().map(|s| s.as_str()).collect();

    group.bench_function("ascii_char_count", |b| {
        b.iter(|| {
            black_box(vectorized_char_count(
                black_box(&ascii_refs),
                black_box('t'),
            ))
        });
    });

    group.bench_function("unicode_char_count", |b| {
        b.iter(|| {
            black_box(vectorized_char_count(
                black_box(&unicode_refs),
                black_box('测'),
            ))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_batch_string_contains,
    benchmark_parallel_string_match,
    benchmark_vectorized_char_count,
    benchmark_short_vs_long_patterns,
    benchmark_ascii_vs_unicode,
);
criterion_main!(benches);
