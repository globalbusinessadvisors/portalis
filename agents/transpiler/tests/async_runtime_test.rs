//! Integration tests for WASI async runtime
//!
//! Tests the cross-platform async runtime implementation across:
//! - Native (tokio)
//! - Browser WASM (wasm-bindgen-futures)
//! - WASI (tokio on wasi)

#[cfg(not(target_arch = "wasm32"))]
mod native_tests {
    use portalis_transpiler::wasi_async_runtime::{
        AsyncRuntime, spawn, spawn_blocking, sleep, timeout, yield_now, TaskHandle, AsyncError,
    };
    use std::time::Duration;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

    #[tokio::test]
    async fn test_spawn_basic() {
        let handle = spawn(async {
            42
        });

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_spawn_with_computation() {
        let handle = spawn(async {
            let mut sum = 0;
            for i in 1..=10 {
                sum += i;
            }
            sum
        });

        let result = handle.await.unwrap();
        assert_eq!(result, 55);
    }

    #[tokio::test]
    async fn test_multiple_spawns() {
        let handle1 = spawn(async {
            sleep(Duration::from_millis(50)).await;
            1
        });

        let handle2 = spawn(async {
            sleep(Duration::from_millis(30)).await;
            2
        });

        let handle3 = spawn(async {
            sleep(Duration::from_millis(20)).await;
            3
        });

        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();
        let result3 = handle3.await.unwrap();

        assert_eq!(result1 + result2 + result3, 6);
    }

    #[tokio::test]
    async fn test_sleep() {
        let start = std::time::Instant::now();
        sleep(Duration::from_millis(100)).await;
        let elapsed = start.elapsed();

        assert!(elapsed >= Duration::from_millis(100));
        assert!(elapsed < Duration::from_millis(200)); // Allow some overhead
    }

    #[tokio::test]
    async fn test_timeout_success() {
        let result = timeout(Duration::from_secs(1), async {
            sleep(Duration::from_millis(100)).await;
            42
        }).await;

        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_timeout_failure() {
        let result = timeout(Duration::from_millis(100), async {
            sleep(Duration::from_secs(1)).await;
            42
        }).await;

        assert!(result.is_err());
        match result {
            Err(AsyncError::Timeout(_)) => (),
            _ => panic!("Expected timeout error"),
        }
    }

    #[tokio::test]
    async fn test_yield_now() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let handle = spawn(async move {
            for _ in 0..5 {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                yield_now().await;
            }
        });

        // Give the spawned task a chance to run
        sleep(Duration::from_millis(10)).await;

        handle.await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_block_on() {
        let result = AsyncRuntime::block_on(async {
            sleep(Duration::from_millis(50)).await;
            42
        });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_block_on_with_spawn() {
        let result = AsyncRuntime::block_on(async {
            let handle = spawn(async {
                sleep(Duration::from_millis(50)).await;
                100
            });

            handle.await.unwrap()
        });

        assert_eq!(result, 100);
    }

    #[tokio::test]
    async fn test_spawn_blocking() {
        let handle = spawn_blocking(|| {
            // Simulate CPU-intensive work
            std::thread::sleep(Duration::from_millis(100));
            42
        });

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_task_cancellation() {
        let handle = spawn(async {
            sleep(Duration::from_secs(10)).await;
            42
        });

        // Abort the task
        handle.abort();

        // Task should be cancelled
        let result = handle.await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_concurrent_tasks() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn 10 concurrent tasks
        for _ in 0..10 {
            let counter_clone = counter.clone();
            let handle = spawn(async move {
                sleep(Duration::from_millis(50)).await;
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[tokio::test]
    async fn test_nested_spawns() {
        let handle = spawn(async {
            let inner_handle = spawn(async {
                sleep(Duration::from_millis(50)).await;
                21
            });

            let inner_result = inner_handle.await.unwrap();
            inner_result * 2
        });

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_error_propagation() {
        let handle = spawn(async {
            // Simulate an error
            Result::<i32, &str>::Err("error")?;
            Ok::<i32, &str>(42)
        });

        let result = handle.await;
        assert!(result.is_ok()); // The outer result is Ok
        let inner_result = result.unwrap();
        assert!(inner_result.is_err()); // The inner result is Err
    }

    #[test]
    fn test_runtime_initialization() {
        // Test that runtime can be initialized
        assert!(AsyncRuntime::init().is_ok());

        // Test idempotency
        assert!(AsyncRuntime::init().is_ok());
    }

    #[tokio::test]
    async fn test_sleep_precision() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(50),
            Duration::from_millis(100),
        ];

        for duration in durations {
            let start = std::time::Instant::now();
            sleep(duration).await;
            let elapsed = start.elapsed();

            assert!(elapsed >= duration, "Sleep was too short: {:?} < {:?}", elapsed, duration);
            assert!(elapsed < duration + Duration::from_millis(100), "Sleep was too long: {:?} >= {:?}", elapsed, duration + Duration::from_millis(100));
        }
    }

    #[tokio::test]
    async fn test_timeout_edge_cases() {
        // Timeout exactly at duration
        let result = timeout(Duration::from_millis(100), async {
            sleep(Duration::from_millis(100)).await;
            42
        }).await;
        // This might succeed or fail depending on timing, so we just check it doesn't panic
        let _ = result;

        // Zero timeout
        let result = timeout(Duration::from_millis(0), async {
            sleep(Duration::from_millis(10)).await;
            42
        }).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_spawn_local() {
        use portalis_transpiler::wasi_async_runtime::spawn_local;

        // spawn_local allows non-Send types in the future
        let local_set = tokio::task::LocalSet::new();

        local_set.run_until(async {
            let handle = spawn_local(async {
                42
            });

            let result = handle.await.unwrap();
            assert_eq!(result, 42);
        }).await;
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod browser_tests {
    use portalis_transpiler::wasi_async_runtime::{
        spawn, sleep, timeout, yield_now, AsyncError,
    };
    use std::time::Duration;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_spawn_basic() {
        let handle = spawn(async {
            42
        });

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[wasm_bindgen_test]
    async fn test_sleep() {
        let start = instant::Instant::now();
        sleep(Duration::from_millis(100)).await;
        let elapsed = start.elapsed();

        // Browser timers are less precise
        assert!(elapsed >= Duration::from_millis(90));
    }

    #[wasm_bindgen_test]
    async fn test_timeout_success() {
        let result = timeout(Duration::from_secs(1), async {
            sleep(Duration::from_millis(100)).await;
            42
        }).await;

        assert_eq!(result.unwrap(), 42);
    }

    #[wasm_bindgen_test]
    async fn test_yield_now() {
        yield_now().await;
        // If we get here, yield worked
        assert!(true);
    }

    #[wasm_bindgen_test]
    async fn test_multiple_tasks() {
        let handle1 = spawn(async {
            sleep(Duration::from_millis(50)).await;
            1
        });

        let handle2 = spawn(async {
            sleep(Duration::from_millis(30)).await;
            2
        });

        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();

        assert_eq!(result1 + result2, 3);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasi"))]
mod wasi_tests {
    use portalis_transpiler::wasi_async_runtime::{
        AsyncRuntime, spawn, sleep, timeout, yield_now, AsyncError,
    };
    use std::time::Duration;

    #[tokio::test]
    async fn test_spawn_basic() {
        let handle = spawn(async {
            42
        });

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_sleep() {
        let start = instant::Instant::now();
        sleep(Duration::from_millis(100)).await;
        let elapsed = start.elapsed();

        assert!(elapsed >= Duration::from_millis(100));
    }

    #[test]
    fn test_block_on() {
        let result = AsyncRuntime::block_on(async {
            sleep(Duration::from_millis(50)).await;
            42
        });

        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_timeout() {
        let result = timeout(Duration::from_millis(100), async {
            sleep(Duration::from_secs(1)).await;
            42
        }).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_yield_now() {
        yield_now().await;
        assert!(true);
    }
}
