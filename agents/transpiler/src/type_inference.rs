//! Hindley-Milner Type Inference System
//!
//! Implements classical Hindley-Milner type inference for Python to Rust translation.
//! Provides automatic type inference, polymorphism, and constraint solving.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Type representation in Hindley-Milner system
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    /// Type variable (e.g., 'a, 'b)
    TVar(TypeVar),
    /// Type constructor (e.g., Int, String, Bool)
    TCon(TypeCon),
    /// Type application (e.g., Vec<T>, Option<T>)
    TApp(Box<Type>, Box<Type>),
    /// Function type (a -> b)
    TFun(Box<Type>, Box<Type>),
}

/// Type variable with unique identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeVar {
    pub name: String,
    pub id: usize,
}

/// Type constructor (base types)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeCon {
    pub name: String,
}

/// Type scheme for polymorphic types (∀a. a -> a)
#[derive(Debug, Clone)]
pub struct Scheme {
    /// Quantified type variables
    pub vars: Vec<TypeVar>,
    /// Body type
    pub ty: Type,
}

/// Substitution: mapping from type variables to types
#[derive(Debug, Clone, Default)]
pub struct Subst {
    map: HashMap<TypeVar, Type>,
}

/// Type environment: maps variables to type schemes
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    env: HashMap<String, Scheme>,
}

/// Type inference error
#[derive(Debug, Clone)]
pub enum TypeError {
    UnificationError(Type, Type),
    OccursCheck(TypeVar, Type),
    UnboundVariable(String),
    InfiniteType(TypeVar, Type),
    ConstraintError(String),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::TVar(tv) => write!(f, "{}", tv.name),
            Type::TCon(tc) => write!(f, "{}", tc.name),
            Type::TApp(t1, t2) => write!(f, "{}<{}>", t1, t2),
            Type::TFun(t1, t2) => write!(f, "({} -> {})", t1, t2),
        }
    }
}

impl Type {
    /// Create a type variable
    pub fn var(name: impl Into<String>, id: usize) -> Self {
        Type::TVar(TypeVar {
            name: name.into(),
            id,
        })
    }

    /// Create a type constructor
    pub fn con(name: impl Into<String>) -> Self {
        Type::TCon(TypeCon { name: name.into() })
    }

    /// Create a function type
    pub fn fun(from: Type, to: Type) -> Self {
        Type::TFun(Box::new(from), Box::new(to))
    }

    /// Create a type application
    pub fn app(constructor: Type, arg: Type) -> Self {
        Type::TApp(Box::new(constructor), Box::new(arg))
    }

    /// Get all free type variables
    pub fn ftv(&self) -> HashSet<TypeVar> {
        match self {
            Type::TVar(tv) => {
                let mut set = HashSet::new();
                set.insert(tv.clone());
                set
            }
            Type::TCon(_) => HashSet::new(),
            Type::TApp(t1, t2) => {
                let mut set = t1.ftv();
                set.extend(t2.ftv());
                set
            }
            Type::TFun(t1, t2) => {
                let mut set = t1.ftv();
                set.extend(t2.ftv());
                set
            }
        }
    }

    /// Apply substitution to type
    pub fn apply(&self, subst: &Subst) -> Type {
        match self {
            Type::TVar(tv) => subst.lookup(tv).unwrap_or_else(|| self.clone()),
            Type::TCon(_) => self.clone(),
            Type::TApp(t1, t2) => {
                Type::TApp(Box::new(t1.apply(subst)), Box::new(t2.apply(subst)))
            }
            Type::TFun(t1, t2) => {
                Type::TFun(Box::new(t1.apply(subst)), Box::new(t2.apply(subst)))
            }
        }
    }

    /// Convert to Rust type string
    pub fn to_rust_type(&self) -> String {
        match self {
            Type::TVar(tv) => tv.name.clone(),
            Type::TCon(tc) => match tc.name.as_str() {
                "Int" => "i32".to_string(),
                "Float" => "f64".to_string(),
                "String" => "String".to_string(),
                "Bool" => "bool".to_string(),
                "Unit" => "()".to_string(),
                name => name.to_string(),
            },
            Type::TApp(t1, t2) => {
                let base = t1.to_rust_type();
                let arg = t2.to_rust_type();
                format!("{}<{}>", base, arg)
            }
            Type::TFun(t1, t2) => {
                format!("Fn({}) -> {}", t1.to_rust_type(), t2.to_rust_type())
            }
        }
    }
}

impl Scheme {
    /// Create a monomorphic scheme (no quantified variables)
    pub fn mono(ty: Type) -> Self {
        Scheme {
            vars: vec![],
            ty,
        }
    }

    /// Create a polymorphic scheme
    pub fn poly(vars: Vec<TypeVar>, ty: Type) -> Self {
        Scheme { vars, ty }
    }

    /// Get free type variables in scheme
    pub fn ftv(&self) -> HashSet<TypeVar> {
        let mut ftv = self.ty.ftv();
        for var in &self.vars {
            ftv.remove(var);
        }
        ftv
    }

    /// Apply substitution to scheme
    pub fn apply(&self, subst: &Subst) -> Scheme {
        // Remove quantified variables from substitution
        let mut new_subst = subst.clone();
        for var in &self.vars {
            new_subst.remove(var);
        }
        Scheme {
            vars: self.vars.clone(),
            ty: self.ty.apply(&new_subst),
        }
    }
}

impl Subst {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Create substitution from single binding
    pub fn singleton(var: TypeVar, ty: Type) -> Self {
        let mut map = HashMap::new();
        map.insert(var, ty);
        Self { map }
    }

    /// Lookup type for variable
    pub fn lookup(&self, var: &TypeVar) -> Option<Type> {
        self.map.get(var).cloned()
    }

    /// Remove a binding
    pub fn remove(&mut self, var: &TypeVar) {
        self.map.remove(var);
    }

    /// Compose two substitutions
    pub fn compose(&self, other: &Subst) -> Subst {
        let mut map = HashMap::new();

        // Apply self to other's types
        for (var, ty) in &other.map {
            map.insert(var.clone(), ty.apply(self));
        }

        // Add self's bindings
        for (var, ty) in &self.map {
            if !map.contains_key(var) {
                map.insert(var.clone(), ty.clone());
            }
        }

        Subst { map }
    }

    /// Apply substitution to a type
    pub fn apply_to_type(&self, ty: &Type) -> Type {
        ty.apply(self)
    }
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            env: HashMap::new(),
        }
    }

    /// Insert a binding
    pub fn insert(&mut self, name: impl Into<String>, scheme: Scheme) {
        self.env.insert(name.into(), scheme);
    }

    /// Lookup a variable
    pub fn lookup(&self, name: &str) -> Option<&Scheme> {
        self.env.get(name)
    }

    /// Remove a binding
    pub fn remove(&mut self, name: &str) {
        self.env.remove(name);
    }

    /// Get free type variables in environment
    pub fn ftv(&self) -> HashSet<TypeVar> {
        self.env.values().flat_map(|s| s.ftv()).collect()
    }

    /// Apply substitution to environment
    pub fn apply(&self, subst: &Subst) -> TypeEnv {
        TypeEnv {
            env: self
                .env
                .iter()
                .map(|(k, v)| (k.clone(), v.apply(subst)))
                .collect(),
        }
    }

    /// Extend environment with standard library types
    pub fn with_stdlib() -> Self {
        let mut env = Self::new();

        // Primitive types
        env.insert("int", Scheme::mono(Type::con("Int")));
        env.insert("float", Scheme::mono(Type::con("Float")));
        env.insert("str", Scheme::mono(Type::con("String")));
        env.insert("bool", Scheme::mono(Type::con("Bool")));

        // Common functions
        // len: ∀a. Vec<a> -> Int
        let a = Type::var("a", 0);
        let vec_a = Type::app(Type::con("Vec"), a.clone());
        env.insert("len", Scheme::poly(vec![TypeVar { name: "a".to_string(), id: 0 }], Type::fun(vec_a, Type::con("Int"))));

        env
    }
}

/// Type inference engine
pub struct TypeInference {
    /// Counter for generating fresh type variables
    var_counter: usize,
    /// Type environment
    pub env: TypeEnv,
    /// Collected constraints
    pub constraints: Vec<(Type, Type)>,
}

impl TypeInference {
    pub fn new() -> Self {
        Self {
            var_counter: 0,
            env: TypeEnv::with_stdlib(),
            constraints: Vec::new(),
        }
    }

    /// Generate a fresh type variable
    pub fn fresh_var(&mut self) -> Type {
        let id = self.var_counter;
        self.var_counter += 1;
        Type::var(format!("t{}", id), id)
    }

    /// Unify two types
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<Subst, TypeError> {
        match (t1, t2) {
            // Identical types
            (Type::TCon(tc1), Type::TCon(tc2)) if tc1 == tc2 => Ok(Subst::new()),

            // Type variable on left
            (Type::TVar(tv), t) | (t, Type::TVar(tv)) => self.bind_var(tv, t),

            // Function types
            (Type::TFun(l1, r1), Type::TFun(l2, r2)) => {
                let s1 = self.unify(l1, l2)?;
                let s2 = self.unify(&r1.apply(&s1), &r2.apply(&s1))?;
                Ok(s1.compose(&s2))
            }

            // Type applications
            (Type::TApp(l1, r1), Type::TApp(l2, r2)) => {
                let s1 = self.unify(l1, l2)?;
                let s2 = self.unify(&r1.apply(&s1), &r2.apply(&s1))?;
                Ok(s1.compose(&s2))
            }

            // Mismatch
            _ => Err(TypeError::UnificationError(t1.clone(), t2.clone())),
        }
    }

    /// Bind a type variable to a type
    fn bind_var(&self, tv: &TypeVar, ty: &Type) -> Result<Subst, TypeError> {
        match ty {
            Type::TVar(tv2) if tv == tv2 => Ok(Subst::new()),
            _ if ty.ftv().contains(tv) => Err(TypeError::OccursCheck(tv.clone(), ty.clone())),
            _ => Ok(Subst::singleton(tv.clone(), ty.clone())),
        }
    }

    /// Instantiate a type scheme (replace quantified variables with fresh ones)
    pub fn instantiate(&mut self, scheme: &Scheme) -> Type {
        let fresh_vars: Vec<Type> = scheme.vars.iter().map(|_| self.fresh_var()).collect();

        let mut subst = Subst::new();
        for (var, fresh) in scheme.vars.iter().zip(fresh_vars.iter()) {
            subst.map.insert(var.clone(), fresh.clone());
        }

        scheme.ty.apply(&subst)
    }

    /// Generalize a type (create a type scheme)
    pub fn generalize(&self, ty: &Type) -> Scheme {
        let env_ftv = self.env.ftv();
        let ty_ftv = ty.ftv();
        let vars: Vec<TypeVar> = ty_ftv.difference(&env_ftv).cloned().collect();
        Scheme::poly(vars, ty.clone())
    }

    /// Infer type of an expression
    pub fn infer(&mut self, expr: &Expr) -> Result<(Subst, Type), TypeError> {
        match expr {
            Expr::Var(name) => {
                if let Some(scheme) = self.env.lookup(name).cloned() {
                    let ty = self.instantiate(&scheme);
                    Ok((Subst::new(), ty))
                } else {
                    Err(TypeError::UnboundVariable(name.clone()))
                }
            }

            Expr::Lit(lit) => {
                let ty = match lit {
                    Literal::Int(_) => Type::con("Int"),
                    Literal::Float(_) => Type::con("Float"),
                    Literal::String(_) => Type::con("String"),
                    Literal::Bool(_) => Type::con("Bool"),
                };
                Ok((Subst::new(), ty))
            }

            Expr::Abs(var, body) => {
                let tv = self.fresh_var();
                let mut new_env = self.env.clone();
                new_env.insert(var, Scheme::mono(tv.clone()));

                let old_env = std::mem::replace(&mut self.env, new_env);
                let (s1, t1) = self.infer(body)?;
                self.env = old_env;

                Ok((s1.clone(), Type::fun(tv.apply(&s1), t1)))
            }

            Expr::App(func, arg) => {
                let (s1, t1) = self.infer(func)?;
                let (s2, t2) = self.infer(arg)?;

                let tv = self.fresh_var();
                let s3 = self.unify(&t1.apply(&s2), &Type::fun(t2.clone(), tv.clone()))?;

                Ok((s3.compose(&s2).compose(&s1), tv.apply(&s3)))
            }

            Expr::Let(var, value, body) => {
                let (s1, t1) = self.infer(value)?;
                let mut new_env = self.env.apply(&s1);

                // Generalize the type
                let scheme = self.generalize(&t1);
                new_env.insert(var, scheme);

                let old_env = std::mem::replace(&mut self.env, new_env);
                let (s2, t2) = self.infer(body)?;
                self.env = old_env;

                Ok((s2.compose(&s1), t2))
            }

            Expr::If(cond, then_expr, else_expr) => {
                let (s1, t1) = self.infer(cond)?;
                let s2 = self.unify(&t1, &Type::con("Bool"))?;

                let (s3, t2) = self.infer(then_expr)?;
                let (s4, t3) = self.infer(else_expr)?;

                let s5 = self.unify(&t2.apply(&s4), &t3)?;

                Ok((
                    s5.compose(&s4).compose(&s3).compose(&s2).compose(&s1),
                    t3.apply(&s5),
                ))
            }
        }
    }

    /// Solve collected constraints
    pub fn solve_constraints(&mut self) -> Result<Subst, TypeError> {
        let mut subst = Subst::new();
        let constraints = self.constraints.clone();

        for (t1, t2) in &constraints {
            let s = self.unify(&t1.apply(&subst), &t2.apply(&subst))?;
            subst = s.compose(&subst);
        }

        Ok(subst)
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, t1: Type, t2: Type) {
        self.constraints.push((t1, t2));
    }
}

/// Simple expression AST for type inference
#[derive(Debug, Clone)]
pub enum Expr {
    Var(String),
    Lit(Literal),
    Abs(String, Box<Expr>), // Lambda: λx. e
    App(Box<Expr>, Box<Expr>), // Application: f x
    Let(String, Box<Expr>, Box<Expr>), // Let binding: let x = e1 in e2
    If(Box<Expr>, Box<Expr>, Box<Expr>), // If expression
}

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

impl Default for TypeInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_identical() {
        let mut inf = TypeInference::new();
        let t1 = Type::con("Int");
        let t2 = Type::con("Int");
        let result = inf.unify(&t1, &t2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_unify_var() {
        let mut inf = TypeInference::new();
        let t1 = Type::var("a", 0);
        let t2 = Type::con("Int");
        let subst = inf.unify(&t1, &t2).unwrap();
        assert_eq!(subst.lookup(&TypeVar { name: "a".to_string(), id: 0 }), Some(Type::con("Int")));
    }

    #[test]
    fn test_infer_literal() {
        let mut inf = TypeInference::new();
        let expr = Expr::Lit(Literal::Int(42));
        let (_, ty) = inf.infer(&expr).unwrap();
        assert_eq!(ty, Type::con("Int"));
    }

    #[test]
    fn test_infer_lambda() {
        let mut inf = TypeInference::new();
        // λx. x
        let expr = Expr::Abs("x".to_string(), Box::new(Expr::Var("x".to_string())));
        let (_, ty) = inf.infer(&expr).unwrap();
        // Should be: t0 -> t0
        match ty {
            Type::TFun(from, to) => assert_eq!(from, to),
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_generalization() {
        let inf = TypeInference::new();
        let ty = Type::fun(Type::var("a", 0), Type::var("a", 0));
        let scheme = inf.generalize(&ty);
        assert_eq!(scheme.vars.len(), 1);
    }
}
