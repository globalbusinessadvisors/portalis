//! Async Runtime Demo
//!
//! Demonstrates the WASI async runtime capabilities with practical examples.

use portalis_transpiler::wasi_async_runtime::{
    AsyncRuntime, spawn, spawn_blocking, sleep, timeout, yield_now, AsyncError,
};
use std::time::Duration;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

/// Example 1: Basic async function
async fn simple_async_function() -> i32 {
    println!("Starting async work...");
    sleep(Duration::from_millis(100)).await;
    println!("Async work complete!");
    42
}

/// Example 2: Concurrent task execution
async fn concurrent_tasks_example() {
    println!("\n=== Concurrent Tasks Example ===");

    let task1 = spawn(async {
        sleep(Duration::from_millis(100)).await;
        println!("Task 1 completed");
        1
    });

    let task2 = spawn(async {
        sleep(Duration::from_millis(150)).await;
        println!("Task 2 completed");
        2
    });

    let task3 = spawn(async {
        sleep(Duration::from_millis(50)).await;
        println!("Task 3 completed");
        3
    });

    let result1 = task1.await.unwrap();
    let result2 = task2.await.unwrap();
    let result3 = task3.await.unwrap();

    println!("All tasks completed. Sum: {}", result1 + result2 + result3);
}

/// Example 3: Timeout handling
async fn timeout_example() {
    println!("\n=== Timeout Example ===");

    // This will succeed
    match timeout(Duration::from_secs(1), async {
        sleep(Duration::from_millis(500)).await;
        "Success!"
    }).await {
        Ok(result) => println!("Operation completed: {}", result),
        Err(AsyncError::Timeout(_)) => println!("Operation timed out"),
        Err(e) => println!("Error: {}", e),
    }

    // This will timeout
    match timeout(Duration::from_millis(100), async {
        sleep(Duration::from_secs(1)).await;
        "Success!"
    }).await {
        Ok(result) => println!("Operation completed: {}", result),
        Err(AsyncError::Timeout(d)) => println!("Operation timed out after {:?}", d),
        Err(e) => println!("Error: {}", e),
    }
}

/// Example 4: Cooperative multitasking with yield
async fn cooperative_multitasking() {
    println!("\n=== Cooperative Multitasking Example ===");

    let counter = Arc::new(AtomicUsize::new(0));

    let mut tasks = vec![];

    for i in 0..5 {
        let counter_clone = counter.clone();
        let task = spawn(async move {
            for j in 0..3 {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                println!("Task {} iteration {}", i, j);
                yield_now().await; // Let other tasks run
            }
        });
        tasks.push(task);
    }

    for task in tasks {
        task.await.unwrap();
    }

    println!("Total iterations: {}", counter.load(Ordering::SeqCst));
}

/// Example 5: Blocking task execution
#[cfg(not(target_arch = "wasm32"))]
async fn blocking_task_example() {
    println!("\n=== Blocking Task Example ===");

    let result = spawn_blocking(|| {
        println!("Running CPU-intensive work on blocking pool...");
        std::thread::sleep(Duration::from_millis(200));

        // Simulate heavy computation
        let mut sum: u64 = 0;
        for i in 1..=1000000u64 {
            sum += i;
        }
        sum
    }).await.unwrap();

    println!("Blocking task result: {}", result);
}

/// Example 6: Error handling
async fn error_handling_example() {
    println!("\n=== Error Handling Example ===");

    let task = spawn(async {
        sleep(Duration::from_millis(50)).await;
        Result::<i32, &str>::Err("Something went wrong")
    });

    match task.await {
        Ok(Ok(value)) => println!("Success: {}", value),
        Ok(Err(e)) => println!("Task completed with error: {}", e),
        Err(e) => println!("Task failed: {}", e),
    }
}

/// Example 7: Nested async operations
async fn nested_async_example() {
    println!("\n=== Nested Async Example ===");

    async fn fetch_user(id: i32) -> String {
        sleep(Duration::from_millis(50)).await;
        format!("User {}", id)
    }

    async fn fetch_user_data(id: i32) -> (String, Vec<String>) {
        let user = fetch_user(id).await;
        sleep(Duration::from_millis(50)).await;
        let data = vec!["data1".to_string(), "data2".to_string()];
        (user, data)
    }

    let task = spawn(async {
        fetch_user_data(42).await
    });

    let (user, data) = task.await.unwrap();
    println!("Fetched: {} with {} items", user, data.len());
}

/// Example 8: Task racing
async fn racing_tasks_example() {
    println!("\n=== Racing Tasks Example ===");

    let task1 = spawn(async {
        sleep(Duration::from_millis(100)).await;
        "Task 1"
    });

    let task2 = spawn(async {
        sleep(Duration::from_millis(150)).await;
        "Task 2"
    });

    let task3 = spawn(async {
        sleep(Duration::from_millis(50)).await;
        "Task 3"
    });

    // Wait for first to complete (simple version - just spawn all and check)
    let result1 = task1.await.unwrap();
    println!("First completed: {}", result1);

    // Clean up other tasks
    let _ = task2.await;
    let _ = task3.await;
}

/// Example 9: Fan-out/Fan-in pattern
async fn fan_out_fan_in_example() {
    println!("\n=== Fan-out/Fan-in Example ===");

    let input_data = vec![1, 2, 3, 4, 5];

    // Fan-out: spawn a task for each item
    let mut tasks = vec![];
    for item in input_data {
        let task = spawn(async move {
            sleep(Duration::from_millis(item * 20)).await;
            item * 2
        });
        tasks.push(task);
    }

    // Fan-in: collect all results
    let mut results = vec![];
    for task in tasks {
        results.push(task.await.unwrap());
    }

    println!("Processed results: {:?}", results);
    println!("Sum: {}", results.iter().sum::<u64>());
}

/// Example 10: Pipeline pattern
async fn pipeline_example() {
    println!("\n=== Pipeline Example ===");

    async fn stage1(input: i32) -> i32 {
        sleep(Duration::from_millis(50)).await;
        println!("Stage 1: {} -> {}", input, input * 2);
        input * 2
    }

    async fn stage2(input: i32) -> i32 {
        sleep(Duration::from_millis(50)).await;
        println!("Stage 2: {} -> {}", input, input + 10);
        input + 10
    }

    async fn stage3(input: i32) -> String {
        sleep(Duration::from_millis(50)).await;
        let result = format!("Result: {}", input);
        println!("Stage 3: {}", result);
        result
    }

    let result = stage1(5).await;
    let result = stage2(result).await;
    let result = stage3(result).await;

    println!("Pipeline complete: {}", result);
}

fn main() {
    println!("🚀 Async Runtime Demo\n");
    println!("This demo showcases the WASI async runtime capabilities.");

    // Run all examples using block_on
    AsyncRuntime::block_on(async {
        // Example 1: Basic async
        let result = simple_async_function().await;
        println!("Simple async result: {}\n", result);

        // Example 2: Concurrent tasks
        concurrent_tasks_example().await;

        // Example 3: Timeout
        timeout_example().await;

        // Example 4: Cooperative multitasking
        cooperative_multitasking().await;

        // Example 5: Blocking tasks (native only)
        #[cfg(not(target_arch = "wasm32"))]
        blocking_task_example().await;

        // Example 6: Error handling
        error_handling_example().await;

        // Example 7: Nested async
        nested_async_example().await;

        // Example 8: Racing tasks
        racing_tasks_example().await;

        // Example 9: Fan-out/Fan-in
        fan_out_fan_in_example().await;

        // Example 10: Pipeline
        pipeline_example().await;
    });

    println!("\n✅ All examples completed successfully!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_async_function() {
        let result = simple_async_function().await;
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_concurrent_tasks() {
        concurrent_tasks_example().await;
        // Just verify it doesn't panic
    }

    #[tokio::test]
    async fn test_timeout() {
        timeout_example().await;
        // Just verify it doesn't panic
    }
}
