//! Threading and Multiprocessing Translation Examples
//!
//! Demonstrates how Python threading and multiprocessing are translated to
//! Rust using std::thread, rayon, crossbeam, and wasi_threading primitives.

use portalis_transpiler::threading_translator::{MultiprocessingMapper, ThreadingTranslator};

fn main() {
    println!("=== Python Threading/Multiprocessing → Rust Translation Examples ===\n");

    // Example 1: Basic thread creation
    example_basic_thread();

    // Example 2: Thread with arguments
    example_thread_with_args();

    // Example 3: Mutex/Lock
    example_mutex();

    // Example 4: Queue (producer-consumer)
    example_queue();

    // Example 5: Event synchronization
    example_event();

    // Example 6: Barrier synchronization
    example_barrier();

    // Example 7: Multiprocessing Pool
    example_multiprocessing_pool();

    // Example 8: Advanced patterns
    example_advanced_patterns();
}

fn example_basic_thread() {
    println!("## Example 1: Basic Thread Creation\n");
    println!("Python:");
    println!(r#"
import threading

def worker():
    print("Worker thread running")
    return 42

t = threading.Thread(target=worker)
t.start()
result = t.join()
"#);

    let mut translator = ThreadingTranslator::new();
    let thread_code = translator.translate_thread_creation("worker", &[], false);

    println!("\nRust:");
    println!("{}", thread_code);
    println!("\nlet result = handle.join().unwrap();");

    println!("\nNote: Rust threads return Result, Python threads don't return values directly");
    println!("\n{}\n", "=".repeat(80));
}

fn example_thread_with_args() {
    println!("## Example 2: Thread with Arguments\n");
    println!("Python:");
    println!(r#"
import threading

def worker(name, count):
    for i in range(count):
        print(f"{{name}}: {{i}}")

t = threading.Thread(target=worker, args=("Worker-1", 5))
t.start()
t.join()
"#);

    let mut translator = ThreadingTranslator::new();
    let thread_code = translator.translate_thread_creation("worker", &["name".to_string(), "count".to_string()], false);

    println!("\nRust:");
    println!("{}", thread_code);
    println!("handle.join().unwrap();");

    println!("\nFeatures:");
    println!("- Arguments cloned before move");
    println!("- Closure captures moved variables");
    println!("- join() waits for completion");

    println!("\n{}\n", "=".repeat(80));
}

fn example_mutex() {
    println!("## Example 3: Mutex/Lock Synchronization\n");
    println!("Python:");
    println!(r#"
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    with lock:
        counter += 1

threads = []
for _ in range(10):
    t = threading.Thread(target=increment)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Counter: {{counter}}")
"#);

    let mut translator = ThreadingTranslator::new();
    let lock = translator.translate_lock();
    let usage = translator.translate_lock_usage("counter", "    *counter += 1;");

    println!("\nRust:");
    println!("use std::sync::{{Arc, Mutex}};");
    println!();
    println!("let counter = Arc::new({});", lock.replace("()", "0"));
    println!("let mut handles = vec![];");
    println!();
    println!("for _ in 0..10 {{");
    println!("    let counter = counter.clone();");
    println!("    let handle = std::thread::spawn(move || {{");
    println!("{}", usage);
    println!("    }});");
    println!("    handles.push(handle);");
    println!("}}");
    println!();
    println!("for handle in handles {{");
    println!("    handle.join().unwrap();");
    println!("}}");
    println!();
    println!("println!(\"Counter: {{}}\", *counter.lock().unwrap());");

    println!("\nKey differences:");
    println!("- Rust uses Arc for shared ownership");
    println!("- RAII: lock released automatically at scope end");
    println!("- No global variables - explicit sharing");

    println!("\n{}\n", "=".repeat(80));
}

fn example_queue() {
    println!("## Example 4: Queue (Producer-Consumer)\n");
    println!("Python:");
    println!(r#"
import threading
from queue import Queue

def producer(q, items):
    for item in items:
        q.put(item)
    q.put(None)  # Sentinel

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Processed: {{item}}")

q = Queue()
p = threading.Thread(target=producer, args=(q, [1, 2, 3, 4, 5]))
c = threading.Thread(target=consumer, args=(q,))

p.start()
c.start()
p.join()
c.join()
"#);

    let mut translator = ThreadingTranslator::new();
    let queue = translator.translate_queue();

    println!("\nRust:");
    println!("use crossbeam::channel::unbounded;");
    println!();
    println!("{}", queue);
    println!();
    println!("// Producer");
    println!("let tx_clone = tx.clone();");
    println!("let producer = std::thread::spawn(move || {{");
    println!("    for item in vec![1, 2, 3, 4, 5] {{");
    println!("        tx_clone.send(item).unwrap();");
    println!("    }}");
    println!("}});");
    println!();
    println!("// Consumer");
    println!("let consumer = std::thread::spawn(move || {{");
    println!("    while let Ok(item) = rx.recv() {{");
    println!("        println!(\"Processed: {{}}\", item);");
    println!("    }}");
    println!("}});");
    println!();
    println!("producer.join().unwrap();");
    println!("drop(tx); // Close channel");
    println!("consumer.join().unwrap();");

    println!("\nFeatures:");
    println!("- crossbeam::channel is lock-free");
    println!("- No sentinel needed - channel closes");
    println!("- Backpressure support");

    println!("\n{}\n", "=".repeat(80));
}

fn example_event() {
    println!("## Example 5: Event Synchronization\n");
    println!("Python:");
    println!(r#"
import threading

event = threading.Event()

def waiter():
    print("Waiting for event...")
    event.wait()
    print("Event received!")

def setter():
    time.sleep(1)
    event.set()

t1 = threading.Thread(target=waiter)
t2 = threading.Thread(target=setter)

t1.start()
t2.start()
t1.join()
t2.join()
"#);

    let mut translator = ThreadingTranslator::new();
    let event = translator.translate_event();
    let wait = translator.translate_event_wait("event");
    let set = translator.translate_event_set("event");

    println!("\nRust:");
    println!("use std::sync::{{Arc, Condvar, Mutex}};");
    println!();
    println!("let event = {};", event);
    println!();
    println!("// Waiter thread");
    println!("let event_clone = event.clone();");
    println!("let waiter = std::thread::spawn(move || {{");
    println!("    println!(\"Waiting for event...\");");
    println!("{}", wait);
    println!("    println!(\"Event received!\");");
    println!("}});");
    println!();
    println!("// Setter thread");
    println!("let event_clone = event.clone();");
    println!("let setter = std::thread::spawn(move || {{");
    println!("    std::thread::sleep(std::time::Duration::from_secs(1));");
    println!("{}", set);
    println!("}});");
    println!();
    println!("waiter.join().unwrap();");
    println!("setter.join().unwrap();");

    println!("\nImplementation:");
    println!("- Event = (Mutex<bool>, Condvar)");
    println!("- wait() uses condvar.wait()");
    println!("- set() uses condvar.notify_all()");

    println!("\n{}\n", "=".repeat(80));
}

fn example_barrier() {
    println!("## Example 6: Barrier Synchronization\n");
    println!("Python:");
    println!(r#"
import threading

barrier = threading.Barrier(3)

def worker(i):
    print(f"Thread {{i}} before barrier")
    barrier.wait()
    print(f"Thread {{i}} after barrier")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
"#);

    let mut translator = ThreadingTranslator::new();
    let barrier = translator.translate_barrier(3);
    let wait = translator.translate_barrier_wait("barrier");

    println!("\nRust:");
    println!("use std::sync::{{Arc, Barrier}};");
    println!();
    println!("let barrier = {};", barrier);
    println!();
    println!("let handles: Vec<_> = (0..3).map(|i| {{");
    println!("    let barrier = barrier.clone();");
    println!("    std::thread::spawn(move || {{");
    println!("        println!(\"Thread {{}} before barrier\", i);");
    println!("        {}", wait);
    println!("        println!(\"Thread {{}} after barrier\", i);");
    println!("    }})");
    println!("}}).collect();");
    println!();
    println!("for handle in handles {{");
    println!("    handle.join().unwrap();");
    println!("}}");

    println!("\nUse case: Synchronize N threads at a point");

    println!("\n{}\n", "=".repeat(80));
}

fn example_multiprocessing_pool() {
    println!("## Example 7: Multiprocessing Pool\n");
    println!("Python:");
    println!(r#"
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == '__main__':
    with Pool(4) as pool:
        numbers = [1, 2, 3, 4, 5, 6, 7, 8]
        results = pool.map(square, numbers)
        print(results)
"#);

    let pool = MultiprocessingMapper::translate_pool(Some(4));
    let map_code = MultiprocessingMapper::translate_pool_map("square", "numbers");

    println!("\nRust:");
    println!("{}", pool);
    println!();
    println!("fn square(x: &i32) -> i32 {{");
    println!("    x * x");
    println!("}}");
    println!();
    println!("let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8];");
    println!("{}", map_code);
    println!("println!(\"{{:?}}\", results);");

    println!("\nKey points:");
    println!("- Rayon provides data parallelism");
    println!("- Work stealing for load balancing");
    println!("- True multiprocessing → threads in Rust");

    println!("\n{}\n", "=".repeat(80));
}

fn example_advanced_patterns() {
    println!("## Example 8: Advanced Threading Patterns\n");

    println!("### Producer-Consumer Pattern:");
    println!(r#"
use portalis_transpiler::threading_translator::ThreadingTranslator;

// Helper function (auto-generated)
let producer_consumer = ThreadingTranslator::generate_threading_patterns();

// Usage:
producer_consumer(
    2,  // num_producers
    3,  // num_consumers
    |i| format!("Item {{}}", i),  // producer
    |item| println!("Consumed: {{}}", item)  // consumer
);
"#);

    println!("\n### Work Stealing Pattern:");
    println!(r#"
use rayon::prelude::*;

let items = vec![1, 2, 3, 4, 5, 6, 7, 8];
let results = work_stealing_map(items, |x| x * x);
// Automatically distributes work across threads
"#);

    println!("\n### Pipeline Pattern:");
    println!(r#"
let input = vec![1, 2, 3, 4, 5];
let results = pipeline(
    input,
    |x| x * 2,      // Stage 1: double
    |x| x + 1       // Stage 2: add one
);
// Results: [3, 5, 7, 9, 11]
"#);

    println!("\n### Scatter-Gather Pattern:");
    println!(r#"
let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
let results = scatter_gather(
    data,
    |x| expensive_computation(x),
    4  // num_workers
);
// Distributes work, gathers results in order
"#);

    println!("\n{}\n", "=".repeat(80));
}
