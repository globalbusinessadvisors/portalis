//! Day 8-9 Feature Tests: Classes and Object-Oriented Programming
//!
//! Tests for:
//! - Basic class definitions
//! - __init__ method (constructor)
//! - Instance methods with self
//! - Instance attributes
//! - Object creation/instantiation
//! - Attribute access
//! - Simple inheritance

use super::feature_translator::FeatureTranslator;

#[test]
fn test_simple_class_definition() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
class Person:
    pass
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub struct Person"));
    assert!(rust.contains("impl Person"));
}

#[test]
fn test_class_with_init() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub struct Person"));
    assert!(rust.contains("pub name:"));
    assert!(rust.contains("pub age:"));
    assert!(rust.contains("pub fn new("));
    assert!(rust.contains("name:"));
    assert!(rust.contains("age:"));
}

#[test]
fn test_class_with_typed_init() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub struct Person"));
    assert!(rust.contains("pub fn new("));
    assert!(rust.contains("name: String") || rust.contains("name: str"));
    assert!(rust.contains("age: i32") || rust.contains("age: int"));
}

#[test]
fn test_class_with_method() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
class Counter:
    def __init__(self, value):
        self.value = value

    def increment(self):
        self.value += 1
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub struct Counter"));
    assert!(rust.contains("pub value:"));
    assert!(rust.contains("pub fn new("));
    assert!(rust.contains("pub fn increment(&self)"));
    assert!(rust.contains("self.value += 1;"));
}

#[test]
fn test_class_with_return_method() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
class Calculator:
    def __init__(self, value):
        self.value = value

    def double(self):
        return self.value * 2
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub struct Calculator"));
    assert!(rust.contains("pub fn double(&self)"));
    assert!(rust.contains("return self.value * 2;"));
}

#[test]
fn test_class_object_creation() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
class Person:
    def __init__(self, name):
        self.name = name

p = Person("Alice")
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub struct Person"));
    assert!(rust.contains("pub fn new("));
    assert!(rust.contains("Person::new") || rust.contains("Person(\"Alice\")"));
}

#[test]
fn test_attribute_access() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
class Person:
    def __init__(self, name):
        self.name = name

p = Person("Alice")
print(p.name)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("p.name"));
}

#[test]
fn test_class_with_multiple_methods() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub struct Rectangle"));
    assert!(rust.contains("pub width:"));
    assert!(rust.contains("pub height:"));
    assert!(rust.contains("pub fn area(&self)"));
    assert!(rust.contains("pub fn perimeter(&self)"));
    assert!(rust.contains("return self.width * self.height;"));
}

#[test]
fn test_class_with_method_parameters() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
class BankAccount:
    def __init__(self, balance):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        self.balance -= amount
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub struct BankAccount"));
    assert!(rust.contains("pub fn deposit(&self, amount:"));
    assert!(rust.contains("pub fn withdraw(&self, amount:"));
    assert!(rust.contains("self.balance += amount;"));
    assert!(rust.contains("self.balance -= amount;"));
}

#[test]
fn test_complete_class_program() {
    let mut translator = FeatureTranslator::new();
    let python = r#"
class Counter:
    def __init__(self, start):
        self.value = start

    def increment(self):
        self.value += 1

    def get_value(self):
        return self.value

c = Counter(0)
c.increment()
c.increment()
print(c.get_value())
"#;
    let rust = translator.translate(python).unwrap();

    println!("Generated Rust:\n{}", rust);
    assert!(rust.contains("pub struct Counter"));
    assert!(rust.contains("pub fn new("));
    assert!(rust.contains("pub fn increment(&self)"));
    assert!(rust.contains("pub fn get_value(&self)"));
    assert!(rust.contains("Counter::new") || rust.contains("let c"));
    assert!(rust.contains("c.increment()"));
    assert!(rust.contains("c.get_value()"));
}
