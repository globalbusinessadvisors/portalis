//! NumPy to ndarray Translation Examples
//!
//! Demonstrates translation of Python NumPy operations to Rust ndarray,
//! including array creation, operations, linear algebra, and scientific computing.

use portalis_transpiler::numpy_translator::*;

fn main() {
    println!("=== NumPy to ndarray Translation Examples ===\n");

    // Example 1: Array creation
    example_array_creation();

    // Example 2: Element-wise operations
    example_elementwise_operations();

    // Example 3: Reduction operations
    example_reduction_operations();

    // Example 4: Linear algebra
    example_linear_algebra();

    // Example 5: Array manipulation
    example_array_manipulation();

    // Example 6: Mathematical functions
    example_mathematical_functions();

    // Example 7: Indexing and slicing
    example_indexing_slicing();

    // Example 8: Real-world scientific computing
    example_scientific_computing();
}

fn example_array_creation() {
    println!("## Example 1: Array Creation\n");

    let mut translator = NumPyTranslator::new();

    println!("Python NumPy                    →  Rust ndarray");
    println!("{}", "-".repeat(70));

    let examples = vec![
        (ArrayCreation::Array, vec!["1, 2, 3".to_string()], "Create from values"),
        (ArrayCreation::Zeros, vec!["(3, 3)".to_string()], "3x3 zeros matrix"),
        (ArrayCreation::Ones, vec!["(2, 4)".to_string()], "2x4 ones matrix"),
        (ArrayCreation::Arange, vec!["0".to_string(), "10".to_string(), "2".to_string()], "Range 0-10 step 2"),
        (ArrayCreation::Linspace, vec!["0".to_string(), "1".to_string(), "100".to_string()], "100 points 0-1"),
        (ArrayCreation::Eye, vec!["3".to_string()], "3x3 identity"),
        (ArrayCreation::Full, vec!["(2, 2)".to_string(), "5".to_string()], "Fill with 5"),
    ];

    for (creation, args, description) in examples {
        let rust = translator.translate_creation(&creation, &args);
        println!("{:<45} # {}", rust, description);
    }

    println!("\nRandom arrays:");
    let rand_examples = vec![
        (ArrayCreation::Random(RandomType::Rand), vec!["3, 3".to_string()], "Uniform [0,1)"),
        (ArrayCreation::Random(RandomType::Randn), vec!["3, 3".to_string()], "Normal dist"),
    ];

    for (creation, args, description) in rand_examples {
        let rust = translator.translate_creation(&creation, &args);
        println!("{:<45} # {}", rust, description);
    }

    println!("\n{}\n", "=".repeat(80));
}

fn example_elementwise_operations() {
    println!("## Example 2: Element-wise Operations\n");

    let mut translator = NumPyTranslator::new();

    println!("Python NumPy              →  Rust ndarray");
    println!("{}", "-".repeat(60));

    let ops = vec![
        (ElementWiseOp::Add, Some("b"), "a + b", "Element-wise addition"),
        (ElementWiseOp::Mul, Some("2"), "a * 2", "Scalar multiplication"),
        (ElementWiseOp::Pow, Some("2."), "a ** 2", "Squaring"),
        (ElementWiseOp::Sqrt, None, "np.sqrt(a)", "Square root"),
        (ElementWiseOp::Exp, None, "np.exp(a)", "Exponential"),
        (ElementWiseOp::Log, None, "np.log(a)", "Natural log"),
        (ElementWiseOp::Sin, None, "np.sin(a)", "Sine"),
        (ElementWiseOp::Abs, None, "np.abs(a)", "Absolute value"),
    ];

    for (op, operand, python, description) in ops {
        let rust = translator.translate_elementwise(&op, "a", operand);
        println!("{:<25} →  {:<50} # {}", python, rust, description);
    }

    println!("\nBroadcasting:");
    println!("NumPy automatically broadcasts shapes");
    println!("ndarray supports broadcasting with same rules");
    println!("  (3, 1) + (1, 4) → (3, 4)  ✓");

    println!("\n{}\n", "=".repeat(80));
}

fn example_reduction_operations() {
    println!("## Example 3: Reduction Operations\n");

    let mut translator = NumPyTranslator::new();

    println!("Python NumPy              →  Rust ndarray");
    println!("{}", "-".repeat(80));

    let reductions = vec![
        (ReductionOp::Sum, None, "a.sum()", "Sum all elements"),
        (ReductionOp::Sum, Some("0"), "a.sum(axis=0)", "Sum along axis 0"),
        (ReductionOp::Mean, None, "a.mean()", "Mean of all elements"),
        (ReductionOp::Std, None, "a.std()", "Standard deviation"),
        (ReductionOp::Min, None, "a.min()", "Minimum value"),
        (ReductionOp::Max, None, "a.max()", "Maximum value"),
        (ReductionOp::ArgMin, None, "a.argmin()", "Index of minimum"),
        (ReductionOp::ArgMax, None, "a.argmax()", "Index of maximum"),
    ];

    for (op, axis, python, description) in reductions {
        let rust = translator.translate_reduction(&op, "a", axis);
        println!("{:<25} →  {}", python, rust);
        println!("{:>25}    {}", "", description);
    }

    println!("\nAxis parameter:");
    println!("  axis=0: reduce rows (column-wise)");
    println!("  axis=1: reduce columns (row-wise)");
    println!("  axis=None: reduce all (default)");

    println!("\n{}\n", "=".repeat(80));
}

fn example_linear_algebra() {
    println!("## Example 4: Linear Algebra\n");

    let mut translator = NumPyTranslator::new();

    println!("Python NumPy                →  Rust ndarray + ndarray-linalg");
    println!("{}", "-".repeat(70));

    let linalg_ops = vec![
        (LinAlgOp::Dot, vec!["a", "b"], "np.dot(a, b)", "Matrix multiplication"),
        (LinAlgOp::MatMul, vec!["a", "b"], "a @ b", "Matrix multiply (operator)"),
        (LinAlgOp::Transpose, vec!["a"], "a.T", "Transpose"),
        (LinAlgOp::Inv, vec!["a"], "np.linalg.inv(a)", "Matrix inverse"),
        (LinAlgOp::Det, vec!["a"], "np.linalg.det(a)", "Determinant"),
        (LinAlgOp::Eig, vec!["a"], "np.linalg.eig(a)", "Eigenvalues/vectors"),
        (LinAlgOp::Svd, vec!["a"], "np.linalg.svd(a)", "SVD decomposition"),
        (LinAlgOp::Solve, vec!["a", "b"], "np.linalg.solve(a, b)", "Solve Ax=b"),
    ];

    for (op, operands, python, description) in linalg_ops {
        let rust = translator.translate_linalg(&op, &operands.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        println!("{:<30} →  {}", python, rust);
        println!("{:>32}    {}", "", description);
    }

    println!("\nRequired crate:");
    println!("  ndarray-linalg = \"0.16\"  # BLAS/LAPACK backend");

    println!("\n{}\n", "=".repeat(80));
}

fn example_array_manipulation() {
    println!("## Example 5: Array Manipulation\n");

    let mut translator = NumPyTranslator::new();

    println!("Python NumPy                     →  Rust ndarray");
    println!("{}", "-".repeat(70));

    let manip_ops = vec![
        (ManipulationOp::Reshape, "a", vec!["(3, 4)".to_string()], "a.reshape(3, 4)", "Reshape array"),
        (ManipulationOp::Flatten, "a", vec![], "a.flatten()", "Flatten to 1D"),
        (ManipulationOp::Ravel, "a", vec![], "a.ravel()", "Ravel to 1D"),
        (ManipulationOp::ExpandDims, "a", vec!["0".to_string()], "np.expand_dims(a, 0)", "Add dimension"),
    ];

    for (op, array, args, python, description) in manip_ops {
        let rust = translator.translate_manipulation(&op, array, &args);
        println!("{:<35} →  {}", python, rust);
        println!("{:>37}    {}", "", description);
    }

    println!("\nCombining arrays:");
    println!("  np.concatenate([a, b])     →  concatenate(Axis(0), &[a.view(), b.view()])");
    println!("  np.stack([a, b])           →  stack(Axis(0), &[a.view(), b.view()])");
    println!("  np.vstack([a, b])          →  concatenate(Axis(0), &[...])");
    println!("  np.hstack([a, b])          →  concatenate(Axis(1), &[...])");

    println!("\n{}\n", "=".repeat(80));
}

fn example_mathematical_functions() {
    println!("## Example 6: Mathematical Functions\n");

    let mut translator = NumPyTranslator::new();

    println!("Python NumPy                →  Rust ndarray");
    println!("{}", "-".repeat(70));

    let math_ops = vec![
        (MathOp::Clip, "a", vec!["0".to_string(), "1".to_string()], "np.clip(a, 0, 1)", "Clamp values"),
        (MathOp::Round, "a", vec![], "np.round(a)", "Round to nearest"),
        (MathOp::Floor, "a", vec![], "np.floor(a)", "Round down"),
        (MathOp::Ceil, "a", vec![], "np.ceil(a)", "Round up"),
        (MathOp::Sign, "a", vec![], "np.sign(a)", "Sign (-1, 0, 1)"),
    ];

    for (op, array, args, python, description) in math_ops {
        let rust = translator.translate_math(&op, array, &args);
        println!("{:<30} →  {}", python, rust);
        println!("{:>32}    {}", "", description);
    }

    println!("\nConditional operations:");
    println!("  np.where(cond, x, y)       →  Zip::from(&cond).and(&x).and(&y).map_collect(...)");
    println!("  np.maximum(a, b)           →  Zip::from(&a).and(&b).map_collect(|&x, &y| x.max(y))");
    println!("  np.minimum(a, b)           →  Zip::from(&a).and(&b).map_collect(|&x, &y| x.min(y))");

    println!("\n{}\n", "=".repeat(80));
}

fn example_indexing_slicing() {
    println!("## Example 7: Indexing and Slicing\n");

    let translator = NumPyTranslator::new();

    println!("Python NumPy         →  Rust ndarray");
    println!("{}", "-".repeat(60));

    println!("Basic indexing:");
    println!("  a[0]              →  {}", translator.translate_indexing("a", &["0".to_string()]));
    println!("  a[1, 2]           →  {}", translator.translate_indexing("a", &["1".to_string(), "2".to_string()]));

    println!("\nSlicing:");
    println!("  a[1:5]            →  {}", translator.translate_slicing("a", "1..5"));
    println!("  a[:, 1]           →  {}", translator.translate_slicing("a", ".., 1"));
    println!("  a[::2]            →  {}", translator.translate_slicing("a", "..;2"));
    println!("  a[1:5:2]          →  {}", translator.translate_slicing("a", "1..5;2"));

    println!("\nAdvanced indexing:");
    println!("  Boolean indexing:  a[a > 0]");
    println!("  Fancy indexing:    a[[0, 2, 4]]");
    println!("  Multi-dim slice:   a[1:3, 2:4]");

    println!("\nSlicing syntax:");
    println!("  s![..]            - all elements");
    println!("  s![a..b]          - range [a, b)");
    println!("  s![..;step]       - with step");
    println!("  s![.., i]         - all rows, column i");

    println!("\n{}\n", "=".repeat(80));
}

fn example_scientific_computing() {
    println!("## Example 8: Real-world Scientific Computing\n");

    let mut translator = NumPyTranslator::new();

    println!("Example: Linear regression using NumPy/ndarray\n");

    println!("Python NumPy:");
    println!(r#"
import numpy as np

# Generate data
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Normal equation: theta = (X^T X)^-1 X^T y
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f"Slope: {{theta[1][0]:.2f}}")
print(f"Intercept: {{theta[0][0]:.2f}}")
"#);

    println!("\nRust ndarray:");
    println!(r#"
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_linalg::*;

// Generate data
let x = Array::random((100, 1), StandardNormal);
let y = &x * 2. + 1. + Array::random((100, 1), StandardNormal) * 0.1;

// Add bias term
let ones = Array::ones((100, 1));
let x_b = concatenate(Axis(1), &[ones.view(), x.view()]).unwrap();

// Normal equation: theta = (X^T X)^-1 X^T y
let xt_x = x_b.t().dot(&x_b);
let xt_y = x_b.t().dot(&y);
let theta = xt_x.inv().unwrap().dot(&xt_y);

println!("Slope: {{:.2}}", theta[[1, 0]]);
println!("Intercept: {{:.2}}", theta[[0, 0]]);
"#);

    println!("\nKey translation points:");
    println!("  1. Random generation: np.random.randn → Array::random(StandardNormal)");
    println!("  2. Operations: * 2 + 1 → &x * 2. + 1. (element-wise)");
    println!("  3. Concatenate: np.c_[...] → concatenate(Axis(1), &[...])");
    println!("  4. Matrix ops: @ → .dot(), .T → .t()");
    println!("  5. Inverse: np.linalg.inv → .inv().unwrap()");

    println!("\nRequired dependencies:");
    for (name, version) in translator.get_cargo_dependencies() {
        println!("  {} = \"{}\"", name, version);
    }

    println!("\n{}\n", "=".repeat(80));
}
