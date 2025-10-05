//! NumPy to ndarray Translation
//!
//! Translates Python NumPy operations to Rust ndarray crate, supporting:
//! - Array creation and manipulation
//! - Mathematical operations
//! - Linear algebra
//! - Broadcasting and slicing
//! - Statistical functions

use std::collections::HashMap;

/// NumPy array creation function
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrayCreation {
    /// np.array([...])
    Array,
    /// np.zeros(shape)
    Zeros,
    /// np.ones(shape)
    Ones,
    /// np.empty(shape)
    Empty,
    /// np.arange(start, stop, step)
    Arange,
    /// np.linspace(start, stop, num)
    Linspace,
    /// np.eye(n)
    Eye,
    /// np.identity(n)
    Identity,
    /// np.full(shape, fill_value)
    Full,
    /// np.random.rand/randn
    Random(RandomType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RandomType {
    Rand,      // uniform [0, 1)
    Randn,     // normal distribution
    Randint,   // random integers
    Choice,    // random choice
}

/// NumPy operation type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumPyOperation {
    /// Element-wise operations
    ElementWise(ElementWiseOp),
    /// Reduction operations
    Reduction(ReductionOp),
    /// Linear algebra
    LinearAlgebra(LinAlgOp),
    /// Array manipulation
    Manipulation(ManipulationOp),
    /// Mathematical functions
    Mathematical(MathOp),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElementWiseOp {
    Add, Sub, Mul, Div, Pow, Mod,
    Abs, Sqrt, Exp, Log, Sin, Cos, Tan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReductionOp {
    Sum, Mean, Std, Var, Min, Max, Prod,
    Any, All, ArgMin, ArgMax,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinAlgOp {
    Dot, MatMul, Transpose, Inv, Det, Eig, Svd, Solve,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManipulationOp {
    Reshape, Flatten, Ravel, Squeeze, ExpandDims,
    Concatenate, Stack, VSplit, HSplit, Tile, Repeat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MathOp {
    Clip, Round, Floor, Ceil, Sign,
    Where, Select, Maximum, Minimum,
}

/// NumPy to ndarray translator
pub struct NumPyTranslator {
    /// Import statements needed
    imports: Vec<String>,
    /// Type mappings
    dtype_map: HashMap<String, String>,
}

impl NumPyTranslator {
    pub fn new() -> Self {
        let mut dtype_map = HashMap::new();

        // NumPy dtype → Rust type
        dtype_map.insert("int32".to_string(), "i32".to_string());
        dtype_map.insert("int64".to_string(), "i64".to_string());
        dtype_map.insert("float32".to_string(), "f32".to_string());
        dtype_map.insert("float64".to_string(), "f64".to_string());
        dtype_map.insert("bool".to_string(), "bool".to_string());
        dtype_map.insert("uint8".to_string(), "u8".to_string());
        dtype_map.insert("uint32".to_string(), "u32".to_string());
        dtype_map.insert("uint64".to_string(), "u64".to_string());

        Self {
            imports: Vec::new(),
            dtype_map,
        }
    }

    /// Translate array creation
    pub fn translate_creation(&mut self, creation: &ArrayCreation, args: &[String]) -> String {
        self.add_import("use ndarray::prelude::*");

        match creation {
            ArrayCreation::Array => {
                // np.array([1, 2, 3]) → Array::from(vec![1, 2, 3])
                if args.is_empty() {
                    return "Array::from(vec![])".to_string();
                }
                format!("Array::from(vec![{}])", args.join(", "))
            }
            ArrayCreation::Zeros => {
                // np.zeros((3, 3)) → Array::zeros((3, 3))
                self.add_import("use ndarray::Array");
                format!("Array::zeros({})", args.join(", "))
            }
            ArrayCreation::Ones => {
                // np.ones((3, 3)) → Array::ones((3, 3))
                format!("Array::ones({})", args.join(", "))
            }
            ArrayCreation::Empty => {
                // np.empty((3, 3)) → Array::zeros((3, 3)) // uninitialized unsafe
                format!("Array::zeros({})", args.join(", "))
            }
            ArrayCreation::Arange => {
                // np.arange(0, 10, 2) → Array::range(0., 10., 2.)
                self.add_import("use ndarray::Array1");
                match args.len() {
                    1 => format!("Array1::range(0., {}., 1.)", args[0]),
                    2 => format!("Array1::range({}., {}., 1.)", args[0], args[1]),
                    3 => format!("Array1::range({}., {}., {}.)", args[0], args[1], args[2]),
                    _ => "Array1::range(0., 10., 1.)".to_string(),
                }
            }
            ArrayCreation::Linspace => {
                // np.linspace(0, 1, 100) → Array::linspace(0., 1., 100)
                self.add_import("use ndarray::Array1");
                if args.len() >= 3 {
                    format!("Array1::linspace({}., {}., {})", args[0], args[1], args[2])
                } else {
                    "Array1::linspace(0., 1., 50)".to_string()
                }
            }
            ArrayCreation::Eye => {
                // np.eye(3) → Array::eye(3)
                self.add_import("use ndarray::Array2");
                format!("Array2::eye({})", args.get(0).unwrap_or(&"3".to_string()))
            }
            ArrayCreation::Identity => {
                // np.identity(3) → Array::eye(3)
                self.add_import("use ndarray::Array2");
                format!("Array2::eye({})", args.get(0).unwrap_or(&"3".to_string()))
            }
            ArrayCreation::Full => {
                // np.full((3, 3), 5) → Array::from_elem((3, 3), 5)
                if args.len() >= 2 {
                    format!("Array::from_elem({}, {})", args[0], args[1])
                } else {
                    "Array::from_elem((3, 3), 0)".to_string()
                }
            }
            ArrayCreation::Random(rand_type) => {
                self.translate_random(rand_type, args)
            }
        }
    }

    fn translate_random(&mut self, rand_type: &RandomType, args: &[String]) -> String {
        self.add_import("use ndarray_rand::RandomExt");
        self.add_import("use ndarray_rand::rand_distr::Uniform");

        match rand_type {
            RandomType::Rand => {
                // np.random.rand(3, 3) → Array::random((3, 3), Uniform::new(0., 1.))
                let shape = if args.is_empty() {
                    "(3, 3)".to_string()
                } else {
                    format!("({})", args.join(", "))
                };
                format!("Array::random({}, Uniform::new(0., 1.))", shape)
            }
            RandomType::Randn => {
                // np.random.randn(3, 3) → Array::random((3, 3), StandardNormal)
                self.add_import("use ndarray_rand::rand_distr::StandardNormal");
                let shape = if args.is_empty() {
                    "(3, 3)".to_string()
                } else {
                    format!("({})", args.join(", "))
                };
                format!("Array::random({}, StandardNormal)", shape)
            }
            RandomType::Randint => {
                // np.random.randint(0, 10, size=(3, 3))
                if args.len() >= 2 {
                    format!("Array::random((3, 3), Uniform::new({}, {}))", args[0], args[1])
                } else {
                    "Array::random((3, 3), Uniform::new(0, 10))".to_string()
                }
            }
            RandomType::Choice => {
                "/* np.random.choice not directly supported */".to_string()
            }
        }
    }

    /// Translate element-wise operation
    pub fn translate_elementwise(&mut self, op: &ElementWiseOp, array: &str, operand: Option<&str>) -> String {
        match op {
            ElementWiseOp::Add => {
                if let Some(other) = operand {
                    format!("{} + {}", array, other)
                } else {
                    array.to_string()
                }
            }
            ElementWiseOp::Sub => {
                if let Some(other) = operand {
                    format!("{} - {}", array, other)
                } else {
                    array.to_string()
                }
            }
            ElementWiseOp::Mul => {
                if let Some(other) = operand {
                    format!("{} * {}", array, other)
                } else {
                    array.to_string()
                }
            }
            ElementWiseOp::Div => {
                if let Some(other) = operand {
                    format!("{} / {}", array, other)
                } else {
                    array.to_string()
                }
            }
            ElementWiseOp::Pow => {
                if let Some(other) = operand {
                    format!("{}.mapv(|x| x.powf({}))", array, other)
                } else {
                    array.to_string()
                }
            }
            ElementWiseOp::Mod => {
                if let Some(other) = operand {
                    format!("{} % {}", array, other)
                } else {
                    array.to_string()
                }
            }
            ElementWiseOp::Abs => format!("{}.mapv(|x| x.abs())", array),
            ElementWiseOp::Sqrt => format!("{}.mapv(|x| x.sqrt())", array),
            ElementWiseOp::Exp => format!("{}.mapv(|x| x.exp())", array),
            ElementWiseOp::Log => format!("{}.mapv(|x| x.ln())", array),
            ElementWiseOp::Sin => format!("{}.mapv(|x| x.sin())", array),
            ElementWiseOp::Cos => format!("{}.mapv(|x| x.cos())", array),
            ElementWiseOp::Tan => format!("{}.mapv(|x| x.tan())", array),
        }
    }

    /// Translate reduction operation
    pub fn translate_reduction(&mut self, op: &ReductionOp, array: &str, axis: Option<&str>) -> String {
        match op {
            ReductionOp::Sum => {
                if let Some(ax) = axis {
                    format!("{}.sum_axis(Axis({}))", array, ax)
                } else {
                    format!("{}.sum()", array)
                }
            }
            ReductionOp::Mean => {
                if let Some(ax) = axis {
                    format!("{}.mean_axis(Axis({})).unwrap()", array, ax)
                } else {
                    format!("{}.mean().unwrap()", array)
                }
            }
            ReductionOp::Std => {
                if let Some(ax) = axis {
                    format!("{}.std_axis(Axis({}), 0.)", array, ax)
                } else {
                    format!("{}.std(0.)", array)
                }
            }
            ReductionOp::Var => {
                if let Some(ax) = axis {
                    format!("{}.var_axis(Axis({}), 0.)", array, ax)
                } else {
                    format!("{}.var(0.)", array)
                }
            }
            ReductionOp::Min => {
                format!("{}.iter().cloned().fold(f64::INFINITY, f64::min)", array)
            }
            ReductionOp::Max => {
                format!("{}.iter().cloned().fold(f64::NEG_INFINITY, f64::max)", array)
            }
            ReductionOp::Prod => {
                format!("{}.iter().cloned().product::<f64>()", array)
            }
            ReductionOp::Any => {
                format!("{}.iter().any(|&x| x)", array)
            }
            ReductionOp::All => {
                format!("{}.iter().all(|&x| x)", array)
            }
            ReductionOp::ArgMin => {
                format!("{}.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0", array)
            }
            ReductionOp::ArgMax => {
                format!("{}.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0", array)
            }
        }
    }

    /// Translate linear algebra operation
    pub fn translate_linalg(&mut self, op: &LinAlgOp, operands: &[String]) -> String {
        self.add_import("use ndarray_linalg::*");

        match op {
            LinAlgOp::Dot => {
                // np.dot(a, b) → a.dot(&b)
                if operands.len() >= 2 {
                    format!("{}.dot(&{})", operands[0], operands[1])
                } else {
                    "/* missing operand */".to_string()
                }
            }
            LinAlgOp::MatMul => {
                // np.matmul(a, b) or a @ b → a.dot(&b)
                if operands.len() >= 2 {
                    format!("{}.dot(&{})", operands[0], operands[1])
                } else {
                    "/* missing operand */".to_string()
                }
            }
            LinAlgOp::Transpose => {
                // np.transpose(a) or a.T → a.t()
                if !operands.is_empty() {
                    format!("{}.t()", operands[0])
                } else {
                    "/* missing operand */".to_string()
                }
            }
            LinAlgOp::Inv => {
                // np.linalg.inv(a) → a.inv().unwrap()
                if !operands.is_empty() {
                    format!("{}.inv().unwrap()", operands[0])
                } else {
                    "/* missing operand */".to_string()
                }
            }
            LinAlgOp::Det => {
                // np.linalg.det(a) → a.det().unwrap()
                if !operands.is_empty() {
                    format!("{}.det().unwrap()", operands[0])
                } else {
                    "/* missing operand */".to_string()
                }
            }
            LinAlgOp::Eig => {
                // np.linalg.eig(a) → a.eig().unwrap()
                if !operands.is_empty() {
                    format!("{}.eig().unwrap()", operands[0])
                } else {
                    "/* missing operand */".to_string()
                }
            }
            LinAlgOp::Svd => {
                // np.linalg.svd(a) → a.svd(true, true).unwrap()
                if !operands.is_empty() {
                    format!("{}.svd(true, true).unwrap()", operands[0])
                } else {
                    "/* missing operand */".to_string()
                }
            }
            LinAlgOp::Solve => {
                // np.linalg.solve(a, b) → a.solve(&b).unwrap()
                if operands.len() >= 2 {
                    format!("{}.solve(&{}).unwrap()", operands[0], operands[1])
                } else {
                    "/* missing operand */".to_string()
                }
            }
        }
    }

    /// Translate array manipulation
    pub fn translate_manipulation(&mut self, op: &ManipulationOp, array: &str, args: &[String]) -> String {
        match op {
            ManipulationOp::Reshape => {
                // np.reshape(a, (3, 3)) → a.into_shape((3, 3)).unwrap()
                if !args.is_empty() {
                    format!("{}.into_shape({}).unwrap()", array, args[0])
                } else {
                    array.to_string()
                }
            }
            ManipulationOp::Flatten => {
                // np.flatten(a) → a.into_shape((a.len(),)).unwrap()
                format!("{}.into_shape(({}.len(),)).unwrap()", array, array)
            }
            ManipulationOp::Ravel => {
                // np.ravel(a) → similar to flatten
                format!("{}.into_shape(({}.len(),)).unwrap()", array, array)
            }
            ManipulationOp::Squeeze => {
                // np.squeeze(a) → remove dimensions of size 1
                format!("/* squeeze: {} */", array)
            }
            ManipulationOp::ExpandDims => {
                // np.expand_dims(a, axis) → insert new axis
                format!("{}.insert_axis(Axis({}))", array, args.get(0).unwrap_or(&"0".to_string()))
            }
            ManipulationOp::Concatenate => {
                // np.concatenate([a, b], axis=0) → concatenate(Axis(0), &[a.view(), b.view()])
                self.add_import("use ndarray::concatenate");
                format!("concatenate(Axis(0), &[/* arrays */])")
            }
            ManipulationOp::Stack => {
                // np.stack([a, b], axis=0) → stack(Axis(0), &[a.view(), b.view()])
                self.add_import("use ndarray::stack");
                format!("stack(Axis(0), &[/* arrays */])")
            }
            ManipulationOp::VSplit => {
                format!("/* vsplit: {} */", array)
            }
            ManipulationOp::HSplit => {
                format!("/* hsplit: {} */", array)
            }
            ManipulationOp::Tile => {
                format!("/* tile: {} */", array)
            }
            ManipulationOp::Repeat => {
                format!("/* repeat: {} */", array)
            }
        }
    }

    /// Translate mathematical function
    pub fn translate_math(&mut self, op: &MathOp, array: &str, args: &[String]) -> String {
        match op {
            MathOp::Clip => {
                // np.clip(a, min, max) → a.mapv(|x| x.max(min).min(max))
                if args.len() >= 2 {
                    format!("{}.mapv(|x| x.max({}).min({}))", array, args[0], args[1])
                } else {
                    array.to_string()
                }
            }
            MathOp::Round => {
                format!("{}.mapv(|x| x.round())", array)
            }
            MathOp::Floor => {
                format!("{}.mapv(|x| x.floor())", array)
            }
            MathOp::Ceil => {
                format!("{}.mapv(|x| x.ceil())", array)
            }
            MathOp::Sign => {
                format!("{}.mapv(|x| if x > 0. {{ 1. }} else if x < 0. {{ -1. }} else {{ 0. }})", array)
            }
            MathOp::Where => {
                // np.where(condition, x, y) → Zip::from(&condition).and(&x).and(&y).map_collect(...)
                self.add_import("use ndarray::Zip");
                format!("Zip::from(&condition).and(&x).and(&y).map_collect(|&c, &x, &y| if c {{ x }} else {{ y }})")
            }
            MathOp::Select => {
                format!("/* select: {} */", array)
            }
            MathOp::Maximum => {
                // np.maximum(a, b) → Zip::from(&a).and(&b).map_collect(|&x, &y| x.max(y))
                self.add_import("use ndarray::Zip");
                if !args.is_empty() {
                    format!("Zip::from(&{}).and(&{}).map_collect(|&x, &y| x.max(y))", array, args[0])
                } else {
                    array.to_string()
                }
            }
            MathOp::Minimum => {
                // np.minimum(a, b) → similar to maximum
                self.add_import("use ndarray::Zip");
                if !args.is_empty() {
                    format!("Zip::from(&{}).and(&{}).map_collect(|&x, &y| x.min(y))", array, args[0])
                } else {
                    array.to_string()
                }
            }
        }
    }

    /// Translate indexing and slicing
    pub fn translate_indexing(&self, array: &str, indices: &[String]) -> String {
        if indices.is_empty() {
            return array.to_string();
        }

        if indices.len() == 1 {
            // Single index: a[0] → a[[0]]
            format!("{}[[{}]]", array, indices[0])
        } else {
            // Multiple indices: a[0, 1] → a[[0, 1]]
            format!("{}[[{}]]", array, indices.join(", "))
        }
    }

    /// Translate slicing
    pub fn translate_slicing(&self, array: &str, slice_spec: &str) -> String {
        // a[1:5] → a.slice(s![1..5])
        // a[:, 1] → a.slice(s![.., 1])
        format!("{}.slice(s![{}])", array, slice_spec)
    }

    /// Add import if not already present
    fn add_import(&mut self, import: &str) {
        if !self.imports.contains(&import.to_string()) {
            self.imports.push(import.to_string());
        }
    }

    /// Get all required imports
    pub fn get_imports(&self) -> Vec<String> {
        self.imports.clone()
    }

    /// Get Cargo dependencies needed
    pub fn get_cargo_dependencies(&self) -> Vec<(&str, &str)> {
        vec![
            ("ndarray", "0.15"),
            ("ndarray-rand", "0.14"),
            ("ndarray-linalg", "0.16"),
        ]
    }
}

impl Default for NumPyTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation() {
        let mut translator = NumPyTranslator::new();

        let result = translator.translate_creation(&ArrayCreation::Zeros, &["(3, 3)".to_string()]);
        assert!(result.contains("Array::zeros"));

        let result = translator.translate_creation(&ArrayCreation::Arange, &["0".to_string(), "10".to_string()]);
        assert!(result.contains("Array1::range"));
    }

    #[test]
    fn test_element_wise() {
        let mut translator = NumPyTranslator::new();

        let result = translator.translate_elementwise(&ElementWiseOp::Add, "a", Some("b"));
        assert_eq!(result, "a + b");

        let result = translator.translate_elementwise(&ElementWiseOp::Sqrt, "a", None);
        assert!(result.contains("sqrt"));
    }

    #[test]
    fn test_reduction() {
        let mut translator = NumPyTranslator::new();

        let result = translator.translate_reduction(&ReductionOp::Sum, "a", None);
        assert!(result.contains("sum"));

        let result = translator.translate_reduction(&ReductionOp::Mean, "a", Some("0"));
        assert!(result.contains("mean_axis"));
    }

    #[test]
    fn test_linalg() {
        let mut translator = NumPyTranslator::new();

        let result = translator.translate_linalg(&LinAlgOp::Dot, &["a".to_string(), "b".to_string()]);
        assert!(result.contains("dot"));

        let result = translator.translate_linalg(&LinAlgOp::Inv, &["a".to_string()]);
        assert!(result.contains("inv"));
    }
}
