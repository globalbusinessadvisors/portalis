//! Reference Optimization
//!
//! Optimizes generated Rust code to use references efficiently, avoiding
//! unnecessary clones and memory allocations while maintaining safety.

use std::collections::{HashMap, HashSet};

/// Reference usage pattern
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferencePattern {
    /// Value is only read, can use &T
    ReadOnly,
    /// Value is modified, needs &mut T
    Mutable,
    /// Value is moved/consumed, needs T
    Owned,
    /// Value is shared across threads, needs Arc<T>
    Shared,
    /// Value is mutated across threads, needs Arc<Mutex<T>>
    SharedMutable,
}

/// Optimization suggestion
#[derive(Debug, Clone)]
pub struct Optimization {
    pub location: String,
    pub pattern: OptimizationPattern,
    pub original: String,
    pub optimized: String,
    pub explanation: String,
    pub safety_note: Option<String>,
}

/// Type of optimization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationPattern {
    /// Remove unnecessary clone()
    CloneElimination,
    /// Use &str instead of String
    StringSlice,
    /// Use iterator instead of collecting
    IteratorChain,
    /// Use reference in function parameter
    BorrowParameter,
    /// Use slice instead of Vec
    SliceParameter,
    /// Return reference instead of owned value
    ReturnReference,
    /// Use Cow for conditional ownership
    CowOptimization,
    /// Avoid intermediate allocations
    AllocationReduction,
    /// Use smart pointer efficiently
    SmartPointer,
}

/// Variable usage tracking
#[derive(Debug, Clone)]
struct VariableUsage {
    name: String,
    reads: usize,
    writes: usize,
    moved: bool,
    borrowed: bool,
    borrowed_mut: bool,
    escapes_scope: bool,
}

impl VariableUsage {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reads: 0,
            writes: 0,
            moved: false,
            borrowed: false,
            borrowed_mut: false,
            escapes_scope: false,
        }
    }

    fn infer_pattern(&self) -> ReferencePattern {
        if self.moved || self.escapes_scope {
            ReferencePattern::Owned
        } else if self.writes > 0 || self.borrowed_mut {
            ReferencePattern::Mutable
        } else {
            ReferencePattern::ReadOnly
        }
    }
}

/// Reference optimizer
pub struct ReferenceOptimizer {
    /// Variable usage tracking
    usage: HashMap<String, VariableUsage>,
    /// Detected optimizations
    optimizations: Vec<Optimization>,
    /// String literal detection
    string_literals: HashSet<String>,
    /// Function return types
    return_types: HashMap<String, String>,
}

impl ReferenceOptimizer {
    pub fn new() -> Self {
        Self {
            usage: HashMap::new(),
            optimizations: Vec::new(),
            string_literals: HashSet::new(),
            return_types: HashMap::new(),
        }
    }

    /// Track variable read
    pub fn track_read(&mut self, var: &str) {
        self.usage.entry(var.to_string())
            .or_insert_with(|| VariableUsage::new(var))
            .reads += 1;
    }

    /// Track variable write
    pub fn track_write(&mut self, var: &str) {
        self.usage.entry(var.to_string())
            .or_insert_with(|| VariableUsage::new(var))
            .writes += 1;
    }

    /// Track variable move
    pub fn track_move(&mut self, var: &str) {
        self.usage.entry(var.to_string())
            .or_insert_with(|| VariableUsage::new(var))
            .moved = true;
    }

    /// Track variable borrow
    pub fn track_borrow(&mut self, var: &str, mutable: bool) {
        let usage = self.usage.entry(var.to_string())
            .or_insert_with(|| VariableUsage::new(var));

        if mutable {
            usage.borrowed_mut = true;
        } else {
            usage.borrowed = true;
        }
    }

    /// Analyze function parameter and suggest optimization
    pub fn optimize_parameter(&mut self, param_name: &str, param_type: &str, usage_in_body: &str) -> String {
        // Check if parameter is only read
        if !usage_in_body.contains(&format!("{} =", param_name))
            && !usage_in_body.contains(&format!("&mut {}", param_name)) {

            // Suggest reference for non-Copy types
            if param_type == "String" {
                self.add_optimization(
                    param_name,
                    OptimizationPattern::BorrowParameter,
                    format!("{}: String", param_name),
                    format!("{}: &str", param_name),
                    "Parameter is only read, use &str to avoid unnecessary allocation".to_string(),
                );
                return format!("{}: &str", param_name);
            } else if param_type.starts_with("Vec<") {
                self.add_optimization(
                    param_name,
                    OptimizationPattern::SliceParameter,
                    format!("{}: {}", param_name, param_type),
                    format!("{}: &[{}]", param_name, param_type.trim_start_matches("Vec<").trim_end_matches('>')),
                    "Parameter is only read, use slice to avoid cloning".to_string(),
                );
                let inner = param_type.trim_start_matches("Vec<").trim_end_matches('>');
                return format!("{}: &[{}]", param_name, inner);
            }
        }

        format!("{}: {}", param_name, param_type)
    }

    /// Optimize string usage
    pub fn optimize_string(&mut self, var_name: &str, value: &str, is_literal: bool) -> String {
        if is_literal {
            self.string_literals.insert(var_name.to_string());

            // String literal can use &str
            self.add_optimization(
                var_name,
                OptimizationPattern::StringSlice,
                format!("let {}: String = \"{}\".to_string()", var_name, value),
                format!("let {}: &str = \"{}\"", var_name, value),
                "String literal doesn't need allocation, use &str".to_string(),
            );

            format!("let {}: &str = \"{}\"", var_name, value)
        } else {
            format!("let {}: String = {}", var_name, value)
        }
    }

    /// Detect and eliminate unnecessary clones
    pub fn eliminate_clone(&mut self, var_name: &str, expr: &str) -> String {
        // Pattern: variable.clone() when variable is last use
        if expr.ends_with(".clone()") {
            let base_var = expr.trim_end_matches(".clone()");

            if let Some(usage) = self.usage.get(base_var) {
                if usage.reads == 1 && !usage.borrowed && !usage.escapes_scope {
                    self.add_optimization(
                        var_name,
                        OptimizationPattern::CloneElimination,
                        expr.to_string(),
                        base_var.to_string(),
                        format!("Last use of {}, clone unnecessary", base_var),
                    );
                    return base_var.to_string();
                }
            }
        }

        expr.to_string()
    }

    /// Optimize iterator chains to avoid intermediate collections
    pub fn optimize_iterator(&mut self, location: &str, code: &str) -> String {
        // Pattern: .collect::<Vec<_>>().iter()
        if code.contains(".collect::<Vec<_>>().iter()") {
            let optimized = code.replace(".collect::<Vec<_>>().iter()", "");

            self.add_optimization(
                location,
                OptimizationPattern::IteratorChain,
                code.to_string(),
                optimized.clone(),
                "Avoid intermediate Vec allocation by chaining iterators".to_string(),
            );

            return optimized;
        }

        // Pattern: .collect() followed by .into_iter()
        if code.contains(".collect::<Vec<_>>().into_iter()") {
            let optimized = code.replace(".collect::<Vec<_>>().into_iter()", "");

            self.add_optimization(
                location,
                OptimizationPattern::IteratorChain,
                code.to_string(),
                optimized.clone(),
                "Chain iterators instead of collecting intermediate vector".to_string(),
            );

            return optimized;
        }

        code.to_string()
    }

    /// Optimize return type to use reference when possible
    pub fn optimize_return(&mut self, func_name: &str, return_type: &str, returns_local: bool) -> String {
        if returns_local {
            // Can't return reference to local variable
            return return_type.to_string();
        }

        // If returning String that comes from parameter, can return &str
        if return_type == "String" {
            self.add_optimization(
                func_name,
                OptimizationPattern::ReturnReference,
                format!("-> {}", return_type),
                "-> &str".to_string(),
                "Return reference to avoid unnecessary allocation".to_string(),
            );
            // Note: This optimization requires parameter analysis
            self.optimizations.last_mut().unwrap().safety_note = Some(
                "Only valid if returned value has lifetime tied to parameter".to_string()
            );
        }

        return_type.to_string()
    }

    /// Suggest Cow for conditional ownership
    pub fn suggest_cow(&mut self, var_name: &str, sometimes_owned: bool, sometimes_borrowed: bool) -> String {
        if sometimes_owned && sometimes_borrowed {
            self.add_optimization(
                var_name,
                OptimizationPattern::CowOptimization,
                "String".to_string(),
                "Cow<'_, str>".to_string(),
                "Use Cow for conditional ownership - avoids cloning when possible".to_string(),
            );
            "Cow<'_, str>".to_string()
        } else if sometimes_borrowed {
            "&str".to_string()
        } else {
            "String".to_string()
        }
    }

    /// Optimize smart pointer usage
    pub fn optimize_smart_pointer(&mut self, var_name: &str, is_shared: bool, is_mutable: bool, is_threadsafe: bool) -> String {
        let suggested = if is_threadsafe {
            if is_mutable {
                "Arc<Mutex<T>>".to_string()
            } else {
                "Arc<T>".to_string()
            }
        } else if is_shared {
            if is_mutable {
                "RefCell<T>".to_string()
            } else {
                "Rc<T>".to_string()
            }
        } else if is_mutable {
            "Box<T>".to_string()
        } else {
            "T".to_string()
        };

        self.add_optimization(
            var_name,
            OptimizationPattern::SmartPointer,
            "Box<T>".to_string(),
            suggested.clone(),
            self.smart_pointer_rationale(is_shared, is_mutable, is_threadsafe),
        );

        suggested
    }

    fn smart_pointer_rationale(&self, shared: bool, mutable: bool, threadsafe: bool) -> String {
        match (shared, mutable, threadsafe) {
            (false, false, _) => "No sharing needed, use T directly".to_string(),
            (false, true, _) => "Single owner with mutation, use Box<T>".to_string(),
            (true, false, false) => "Shared ownership, immutable, use Rc<T>".to_string(),
            (true, true, false) => "Shared ownership, mutable, use Rc<RefCell<T>>".to_string(),
            (true, false, true) => "Thread-safe shared ownership, use Arc<T>".to_string(),
            (true, true, true) => "Thread-safe shared mutation, use Arc<Mutex<T>>".to_string(),
        }
    }

    /// Add optimization suggestion
    fn add_optimization(
        &mut self,
        location: impl Into<String>,
        pattern: OptimizationPattern,
        original: impl Into<String>,
        optimized: impl Into<String>,
        explanation: impl Into<String>,
    ) {
        self.optimizations.push(Optimization {
            location: location.into(),
            pattern,
            original: original.into(),
            optimized: optimized.into(),
            explanation: explanation.into(),
            safety_note: None,
        });
    }

    /// Get all optimizations found
    pub fn get_optimizations(&self) -> &[Optimization] {
        &self.optimizations
    }

    /// Generate optimization report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Reference Optimization Report ===\n\n");

        if self.optimizations.is_empty() {
            report.push_str("No optimizations found - code is already efficient!\n");
            return report;
        }

        report.push_str(&format!("Found {} optimization opportunities:\n\n", self.optimizations.len()));

        let mut by_pattern: HashMap<OptimizationPattern, Vec<&Optimization>> = HashMap::new();
        for opt in &self.optimizations {
            by_pattern.entry(opt.pattern.clone())
                .or_default()
                .push(opt);
        }

        for (pattern, opts) in by_pattern {
            report.push_str(&format!("## {:?} ({} occurrences)\n\n", pattern, opts.len()));

            for opt in opts {
                report.push_str(&format!("Location: {}\n", opt.location));
                report.push_str(&format!("  Original:  {}\n", opt.original));
                report.push_str(&format!("  Optimized: {}\n", opt.optimized));
                report.push_str(&format!("  Reason: {}\n", opt.explanation));
                if let Some(note) = &opt.safety_note {
                    report.push_str(&format!("  ⚠️  Safety: {}\n", note));
                }
                report.push('\n');
            }
        }

        report
    }

    /// Clear all tracked data
    pub fn clear(&mut self) {
        self.usage.clear();
        self.optimizations.clear();
        self.string_literals.clear();
        self.return_types.clear();
    }
}

impl Default for ReferenceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Common optimization patterns
pub struct OptimizationPatterns;

impl OptimizationPatterns {
    /// Example: Avoiding unnecessary clones
    pub fn clone_elimination_example() -> (&'static str, &'static str) {
        let before = r#"
fn process(data: Vec<i32>) -> Vec<i32> {
    let copy = data.clone();  // Unnecessary!
    copy
}
"#;

        let after = r#"
fn process(data: Vec<i32>) -> Vec<i32> {
    data  // No clone needed, data is consumed
}
"#;

        (before, after)
    }

    /// Example: String slice optimization
    pub fn string_slice_example() -> (&'static str, &'static str) {
        let before = r#"
fn greet(name: String) -> String {
    format!("Hello, {}", name)
}

let msg = greet("Alice".to_string());  // Unnecessary allocation
"#;

        let after = r#"
fn greet(name: &str) -> String {
    format!("Hello, {}", name)
}

let msg = greet("Alice");  // No allocation needed
"#;

        (before, after)
    }

    /// Example: Iterator chain optimization
    pub fn iterator_chain_example() -> (&'static str, &'static str) {
        let before = r#"
let result = data
    .iter()
    .map(|x| x * 2)
    .collect::<Vec<_>>()  // Unnecessary intermediate Vec
    .iter()
    .filter(|x| **x > 10)
    .collect();
"#;

        let after = r#"
let result = data
    .iter()
    .map(|x| x * 2)
    .filter(|x| *x > 10)  // Chain directly
    .collect();
"#;

        (before, after)
    }

    /// Example: Slice parameter
    pub fn slice_parameter_example() -> (&'static str, &'static str) {
        let before = r#"
fn sum(numbers: Vec<i32>) -> i32 {
    numbers.iter().sum()
}

// Caller must clone or lose ownership
let total = sum(vec.clone());
"#;

        let after = r#"
fn sum(numbers: &[i32]) -> i32 {
    numbers.iter().sum()
}

// Caller can pass reference
let total = sum(&vec);
"#;

        (before, after)
    }

    /// Example: Cow optimization
    pub fn cow_example() -> (&'static str, &'static str) {
        let before = r#"
fn process(s: String, uppercase: bool) -> String {
    if uppercase {
        s.to_uppercase()  // Must clone even for owned String
    } else {
        s
    }
}
"#;

        let after = r#"
use std::borrow::Cow;

fn process(s: &str, uppercase: bool) -> Cow<str> {
    if uppercase {
        Cow::Owned(s.to_uppercase())
    } else {
        Cow::Borrowed(s)
    }
}
"#;

        (before, after)
    }

    /// Example: Smart pointer selection
    pub fn smart_pointer_example() -> &'static str {
        r#"
// Single ownership
let data = Box::new(value);

// Shared ownership (single thread)
let data = Rc::new(value);
let shared = Rc::clone(&data);

// Shared ownership (multi-thread)
let data = Arc::new(value);
let shared = Arc::clone(&data);

// Shared mutation (single thread)
let data = Rc::new(RefCell::new(value));
data.borrow_mut().update();

// Shared mutation (multi-thread)
let data = Arc::new(Mutex::new(value));
data.lock().unwrap().update();
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_optimization() {
        let mut optimizer = ReferenceOptimizer::new();

        let optimized = optimizer.optimize_parameter("name", "String", "println!(\"{}\" name)");
        assert_eq!(optimized, "name: &str");

        assert_eq!(optimizer.optimizations.len(), 1);
        assert_eq!(optimizer.optimizations[0].pattern, OptimizationPattern::BorrowParameter);
    }

    #[test]
    fn test_clone_elimination() {
        let mut optimizer = ReferenceOptimizer::new();

        optimizer.usage.insert("data".to_string(), {
            let mut usage = VariableUsage::new("data");
            usage.reads = 1;
            usage
        });

        let result = optimizer.eliminate_clone("result", "data.clone()");
        assert_eq!(result, "data");
        assert_eq!(optimizer.optimizations.len(), 1);
    }

    #[test]
    fn test_iterator_optimization() {
        let mut optimizer = ReferenceOptimizer::new();

        let code = "data.iter().map(|x| x * 2).collect::<Vec<_>>().iter()";
        let optimized = optimizer.optimize_iterator("chain", code);

        assert!(!optimized.contains(".collect::<Vec<_>>().iter()"));
        assert_eq!(optimizer.optimizations.len(), 1);
        assert_eq!(optimizer.optimizations[0].pattern, OptimizationPattern::IteratorChain);
    }

    #[test]
    fn test_smart_pointer_selection() {
        let mut optimizer = ReferenceOptimizer::new();

        let result = optimizer.optimize_smart_pointer("data", true, true, true);
        assert_eq!(result, "Arc<Mutex<T>>");

        let result = optimizer.optimize_smart_pointer("data", true, false, false);
        assert_eq!(result, "Rc<T>");
    }
}
