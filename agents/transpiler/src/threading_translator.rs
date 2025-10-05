//! Python Threading/Multiprocessing to Rust Translation
//!
//! Translates Python threading and multiprocessing patterns to Rust using:
//! - std::thread for basic threading
//! - wasi_threading primitives for cross-platform support
//! - rayon for data parallelism
//! - crossbeam for channels and synchronization

use crate::python_ast::{PyExpr, PyStmt, TypeAnnotation};
use std::collections::HashMap;

/// Main threading translator
pub struct ThreadingTranslator {
    /// Track used threading patterns
    used_patterns: Vec<ThreadingPattern>,
    /// Required imports
    required_imports: HashMap<String, Vec<String>>,
}

/// Threading patterns detected
#[derive(Debug, Clone)]
pub enum ThreadingPattern {
    BasicThread,
    ThreadWithArgs,
    ThreadPool,
    Lock,
    RLock,
    Semaphore,
    Event,
    Condition,
    Queue,
    Process,
    Pool,
    Barrier,
    ThreadLocal,
}

/// Translation strategy for threading constructs
#[derive(Debug, Clone)]
pub enum ThreadingStrategy {
    /// Use std::thread
    StdThread,
    /// Use wasi_threading
    WasiThreading,
    /// Use rayon for data parallelism
    Rayon,
    /// Use crossbeam for channels
    Crossbeam,
}

impl Default for ThreadingTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadingTranslator {
    pub fn new() -> Self {
        Self {
            used_patterns: Vec::new(),
            required_imports: HashMap::new(),
        }
    }

    /// Translate Python threading.Thread to Rust
    pub fn translate_thread_creation(
        &mut self,
        target_func: &str,
        args: &[String],
        daemon: bool,
    ) -> String {
        self.used_patterns.push(ThreadingPattern::BasicThread);
        self.add_import("std::thread", vec![]);

        if args.is_empty() {
            format!(
                r#"let handle = std::thread::spawn(|| {{
    {}()
}});"#,
                target_func
            )
        } else {
            let args_capture = args
                .iter()
                .map(|arg| format!("let {} = {}.clone();", arg, arg))
                .collect::<Vec<_>>()
                .join("\n    ");

            let args_list = args.join(", ");

            format!(
                r#"{{
    {}
    let handle = std::thread::spawn(move || {{
        {}({})
    }});
}}"#,
                args_capture, target_func, args_list
            )
        }
    }

    /// Translate Python threading.Lock to Rust Mutex
    pub fn translate_lock(&mut self) -> String {
        self.used_patterns.push(ThreadingPattern::Lock);
        self.add_import("std::sync", vec!["std::sync::Mutex".to_string()]);

        "Mutex::new(())".to_string()
    }

    /// Translate Python lock.acquire() / lock.release() to RAII
    pub fn translate_lock_usage(&mut self, lock_var: &str, body: &str) -> String {
        format!(
            r#"{{
    let _guard = {}.lock().unwrap();
{}
}}"#,
            lock_var, body
        )
    }

    /// Translate Python threading.RLock to Rust RwLock
    pub fn translate_rlock(&mut self) -> String {
        self.used_patterns.push(ThreadingPattern::RLock);
        self.add_import("std::sync", vec!["std::sync::RwLock".to_string()]);

        "RwLock::new(())".to_string()
    }

    /// Translate Python threading.Semaphore
    pub fn translate_semaphore(&mut self, value: usize) -> String {
        self.used_patterns.push(ThreadingPattern::Semaphore);
        self.add_import("tokio::sync", vec!["tokio::sync::Semaphore".to_string()]);

        format!("Arc::new(Semaphore::new({}))", value)
    }

    /// Translate Python threading.Event
    pub fn translate_event(&mut self) -> String {
        self.used_patterns.push(ThreadingPattern::Event);
        self.add_import("std::sync", vec![
            "std::sync::Arc".to_string(),
            "std::sync::Condvar".to_string(),
            "std::sync::Mutex".to_string(),
        ]);

        r#"Arc::new((Mutex::new(false), Condvar::new()))"#.to_string()
    }

    /// Translate event.wait()
    pub fn translate_event_wait(&mut self, event_var: &str) -> String {
        format!(
            r#"{{
    let (lock, cvar) = &*{};
    let mut started = lock.lock().unwrap();
    while !*started {{
        started = cvar.wait(started).unwrap();
    }}
}}"#,
            event_var
        )
    }

    /// Translate event.set()
    pub fn translate_event_set(&mut self, event_var: &str) -> String {
        format!(
            r#"{{
    let (lock, cvar) = &*{};
    let mut started = lock.lock().unwrap();
    *started = true;
    cvar.notify_all();
}}"#,
            event_var
        )
    }

    /// Translate Python queue.Queue to Rust crossbeam channel
    pub fn translate_queue(&mut self) -> String {
        self.used_patterns.push(ThreadingPattern::Queue);
        self.add_import("crossbeam::channel", vec!["crossbeam::channel::unbounded".to_string()]);

        "let (tx, rx) = unbounded();".to_string()
    }

    /// Translate queue.put()
    pub fn translate_queue_put(&mut self, queue_var: &str, value: &str) -> String {
        format!("{}.send({}).unwrap();", queue_var, value)
    }

    /// Translate queue.get()
    pub fn translate_queue_get(&mut self, queue_var: &str) -> String {
        format!("{}.recv().unwrap()", queue_var)
    }

    /// Translate Python multiprocessing.Process to Rust thread
    pub fn translate_process(&mut self, target_func: &str, args: &[String]) -> String {
        self.used_patterns.push(ThreadingPattern::Process);
        // In Rust, we use threads for processes (true multiprocessing requires different approach)
        self.translate_thread_creation(target_func, args, false)
    }

    /// Translate Python multiprocessing.Pool
    pub fn translate_pool(&mut self, num_workers: usize) -> String {
        self.used_patterns.push(ThreadingPattern::Pool);
        self.add_import("rayon", vec!["rayon::ThreadPoolBuilder".to_string()]);

        format!(
            r#"ThreadPoolBuilder::new()
    .num_threads({})
    .build()
    .unwrap()"#,
            num_workers
        )
    }

    /// Translate pool.map()
    pub fn translate_pool_map(&mut self, pool_var: &str, func: &str, iterable: &str) -> String {
        self.add_import("rayon::prelude", vec!["rayon::prelude::*".to_string()]);

        format!(
            r#"{}.install(|| {{
    {}.par_iter()
        .map(|x| {}(x))
        .collect::<Vec<_>>()
}})"#,
            pool_var, iterable, func
        )
    }

    /// Translate Python threading.Barrier
    pub fn translate_barrier(&mut self, parties: usize) -> String {
        self.used_patterns.push(ThreadingPattern::Barrier);
        self.add_import("std::sync", vec!["std::sync::Barrier".to_string()]);

        format!("Arc::new(Barrier::new({}))", parties)
    }

    /// Translate barrier.wait()
    pub fn translate_barrier_wait(&mut self, barrier_var: &str) -> String {
        format!("{}.wait();", barrier_var)
    }

    /// Translate Python threading.local()
    pub fn translate_thread_local(&mut self) -> String {
        self.used_patterns.push(ThreadingPattern::ThreadLocal);

        r#"thread_local! {
    static LOCAL_DATA: RefCell<HashMap<String, String>> = RefCell::new(HashMap::new());
}"#
        .to_string()
    }

    /// Generate comprehensive threading patterns
    pub fn generate_threading_patterns() -> String {
        r#"// Threading pattern helpers

/// Producer-Consumer pattern
pub fn producer_consumer<T: Send + 'static>(
    num_producers: usize,
    num_consumers: usize,
    producer_fn: impl Fn(usize) -> T + Send + Sync + 'static,
    consumer_fn: impl Fn(T) + Send + Sync + 'static,
) where
    T: Clone,
{
    use crossbeam::channel::unbounded;
    use std::sync::Arc;

    let (tx, rx) = unbounded();

    // Spawn producers
    let producer_fn = Arc::new(producer_fn);
    for i in 0..num_producers {
        let tx = tx.clone();
        let producer_fn = producer_fn.clone();
        std::thread::spawn(move || {
            let item = producer_fn(i);
            tx.send(item).unwrap();
        });
    }
    drop(tx);

    // Spawn consumers
    let consumer_fn = Arc::new(consumer_fn);
    let mut handles = vec![];
    for _ in 0..num_consumers {
        let rx = rx.clone();
        let consumer_fn = consumer_fn.clone();
        let handle = std::thread::spawn(move || {
            while let Ok(item) = rx.recv() {
                consumer_fn(item);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Work stealing pattern using rayon
pub fn work_stealing_map<T, R>(items: Vec<T>, f: impl Fn(&T) -> R + Sync) -> Vec<R>
where
    T: Send + Sync,
    R: Send,
{
    use rayon::prelude::*;
    items.par_iter().map(f).collect()
}

/// Pipeline pattern with channels
pub fn pipeline<T: Send + 'static, U: Send + 'static, V: Send + 'static>(
    input: Vec<T>,
    stage1: impl Fn(T) -> U + Send + 'static,
    stage2: impl Fn(U) -> V + Send + 'static,
) -> Vec<V> {
    use crossbeam::channel::unbounded;

    let (tx1, rx1) = unbounded();
    let (tx2, rx2) = unbounded();

    // Stage 1
    std::thread::spawn(move || {
        for item in input {
            tx1.send(stage1(item)).unwrap();
        }
    });

    // Stage 2
    std::thread::spawn(move || {
        while let Ok(item) = rx1.recv() {
            tx2.send(stage2(item)).unwrap();
        }
    });

    // Collect results
    rx2.iter().collect()
}

/// Scatter-gather pattern
pub fn scatter_gather<T, R>(
    items: Vec<T>,
    worker_fn: impl Fn(T) -> R + Send + Sync + 'static,
    num_workers: usize,
) -> Vec<R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    use crossbeam::channel::unbounded;
    use std::sync::Arc;

    let (work_tx, work_rx) = unbounded();
    let (result_tx, result_rx) = unbounded();

    // Send work
    for item in items {
        work_tx.send(item).unwrap();
    }
    drop(work_tx);

    // Spawn workers
    let worker_fn = Arc::new(worker_fn);
    let mut handles = vec![];
    for _ in 0..num_workers {
        let work_rx = work_rx.clone();
        let result_tx = result_tx.clone();
        let worker_fn = worker_fn.clone();

        let handle = std::thread::spawn(move || {
            while let Ok(item) = work_rx.recv() {
                let result = worker_fn(item);
                result_tx.send(result).unwrap();
            }
        });
        handles.push(handle);
    }
    drop(result_tx);

    // Gather results
    let results = result_rx.iter().collect();

    for handle in handles {
        handle.join().unwrap();
    }

    results
}
"#
        .to_string()
    }

    // Helper methods

    fn add_import(&mut self, crate_name: &str, items: Vec<String>) {
        self.required_imports
            .entry(crate_name.to_string())
            .or_insert_with(Vec::new)
            .extend(items);
    }

    pub fn generate_imports(&self) -> String {
        let mut imports = Vec::new();

        for (crate_name, items) in &self.required_imports {
            if items.is_empty() {
                imports.push(format!("use {};", crate_name));
            } else {
                for item in items {
                    imports.push(format!("use {};", item));
                }
            }
        }

        // Add Arc for thread safety
        if !imports.is_empty() {
            imports.insert(0, "use std::sync::Arc;".to_string());
        }

        imports.join("\n")
    }

    pub fn get_used_patterns(&self) -> &[ThreadingPattern] {
        &self.used_patterns
    }
}

/// Multiprocessing mapper for Python multiprocessing module
pub struct MultiprocessingMapper;

impl MultiprocessingMapper {
    /// Translate multiprocessing.Pool to rayon ThreadPool
    pub fn translate_pool(num_processes: Option<usize>) -> String {
        let workers = num_processes
            .map(|n| n.to_string())
            .unwrap_or_else(|| "num_cpus::get()".to_string());

        format!(
            r#"use rayon::ThreadPoolBuilder;

let pool = ThreadPoolBuilder::new()
    .num_threads({})
    .build()
    .unwrap();"#,
            workers
        )
    }

    /// Translate Pool.map to parallel iterator
    pub fn translate_pool_map(func: &str, iterable: &str) -> String {
        format!(
            r#"use rayon::prelude::*;

let results: Vec<_> = {}
    .par_iter()
    .map(|x| {}(x))
    .collect();"#,
            iterable, func
        )
    }

    /// Translate Pool.starmap
    pub fn translate_starmap(func: &str, iterable: &str) -> String {
        format!(
            r#"use rayon::prelude::*;

let results: Vec<_> = {}
    .par_iter()
    .map(|(args)| {{
        {}(args)
    }})
    .collect();"#,
            iterable, func
        )
    }

    /// Translate multiprocessing.Queue to crossbeam channel
    pub fn translate_queue() -> String {
        r#"use crossbeam::channel::unbounded;

let (tx, rx) = unbounded();"#
            .to_string()
    }

    /// Translate multiprocessing.Pipe
    pub fn translate_pipe() -> String {
        r#"use crossbeam::channel::unbounded;

let (parent_tx, child_rx) = unbounded();
let (child_tx, parent_rx) = unbounded();"#
            .to_string()
    }

    /// Translate multiprocessing.Manager (shared memory)
    pub fn translate_manager() -> String {
        r#"use std::sync::{Arc, Mutex};

// Shared data structure
let shared_data = Arc::new(Mutex::new(HashMap::new()));"#
            .to_string()
    }

    /// Translate multiprocessing.Lock
    pub fn translate_lock() -> String {
        r#"use std::sync::{Arc, Mutex};

let lock = Arc::new(Mutex::new(()));"#
            .to_string()
    }

    /// Translate multiprocessing.Value (shared value)
    pub fn translate_value(type_hint: &str, initial: &str) -> String {
        format!(
            r#"use std::sync::{{Arc, Mutex}};

let value = Arc::new(Mutex::new({} as {}));"#,
            initial, type_hint
        )
    }

    /// Translate multiprocessing.Array (shared array)
    pub fn translate_array(type_hint: &str, size: usize) -> String {
        format!(
            r#"use std::sync::{{Arc, Mutex}};

let array = Arc::new(Mutex::new(vec![{} as {}; {}]));"#,
            "Default::default()", type_hint, size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_creation() {
        let mut translator = ThreadingTranslator::new();
        let code = translator.translate_thread_creation("worker_func", &[], false);
        assert!(code.contains("std::thread::spawn"));
        assert!(code.contains("worker_func()"));
    }

    #[test]
    fn test_lock_translation() {
        let mut translator = ThreadingTranslator::new();
        let lock = translator.translate_lock();
        assert!(lock.contains("Mutex::new"));
    }

    #[test]
    fn test_queue_translation() {
        let mut translator = ThreadingTranslator::new();
        let queue = translator.translate_queue();
        assert!(queue.contains("unbounded"));
    }

    #[test]
    fn test_pool_translation() {
        let mut translator = ThreadingTranslator::new();
        let pool = translator.translate_pool(4);
        assert!(pool.contains("ThreadPoolBuilder"));
        assert!(pool.contains("num_threads(4)"));
    }

    #[test]
    fn test_import_generation() {
        let mut translator = ThreadingTranslator::new();
        translator.translate_lock();
        translator.translate_queue();

        let imports = translator.generate_imports();
        assert!(imports.contains("std::sync::Mutex"));
        assert!(imports.contains("crossbeam::channel"));
    }
}
