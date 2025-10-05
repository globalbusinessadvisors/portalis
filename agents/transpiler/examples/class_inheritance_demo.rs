//! Class Inheritance Translation Examples
//!
//! Demonstrates how Python class inheritance is translated to Rust using
//! traits and composition.

use portalis_transpiler::class_inheritance::*;

fn main() {
    println!("=== Python Class Inheritance → Rust Translation Examples ===\n");

    // Example 1: Simple class without inheritance
    example_simple_class();

    // Example 2: Single inheritance
    example_single_inheritance();

    // Example 3: Abstract base class (trait)
    example_abstract_base_class();

    // Example 4: Multiple inheritance
    example_multiple_inheritance();

    // Example 5: Method Resolution Order (MRO)
    example_mro();
}

fn example_simple_class() {
    println!("## Example 1: Simple Class (No Inheritance)\n");
    println!("Python:");
    println!(r#"
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
"#);

    let mut translator = ClassInheritanceTranslator::new();

    let class_info = ClassInfo {
        name: "Point".to_string(),
        bases: vec![],
        methods: vec![
            MethodInfo {
                name: "new".to_string(),
                is_abstract: false,
                params: vec!["x".to_string(), "y".to_string()],
                return_type: None,
            },
            MethodInfo {
                name: "distance_from_origin".to_string(),
                is_abstract: false,
                params: vec!["self".to_string()],
                return_type: Some("float".to_string()),
            },
        ],
        attributes: vec![
            AttributeInfo {
                name: "x".to_string(),
                type_hint: Some("int".to_string()),
            },
            AttributeInfo {
                name: "y".to_string(),
                type_hint: Some("int".to_string()),
            },
        ],
        is_abstract: false,
    };

    translator.register_class(class_info.clone());

    match translator.translate_class("Point") {
        Ok(rust_code) => {
            println!("\nRust:");
            println!("{}", rust_code);
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!("\n{}\n", "=".repeat(80));
}

fn example_single_inheritance() {
    println!("## Example 2: Single Inheritance\n");
    println!("Python:");
    println!(r#"
class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return "Some sound"

class Dog(Animal):
    def __init__(self, name: str, breed: str):
        super().__init__(name)
        self.breed = breed

    def bark(self) -> str:
        return "Woof!"
"#);

    let mut translator = ClassInheritanceTranslator::new();

    let base = ClassInfo {
        name: "Animal".to_string(),
        bases: vec![],
        methods: vec![
            MethodInfo {
                name: "new".to_string(),
                is_abstract: false,
                params: vec!["name".to_string()],
                return_type: None,
            },
            MethodInfo {
                name: "speak".to_string(),
                is_abstract: false,
                params: vec!["self".to_string()],
                return_type: Some("str".to_string()),
            },
        ],
        attributes: vec![
            AttributeInfo {
                name: "name".to_string(),
                type_hint: Some("str".to_string()),
            },
        ],
        is_abstract: false,
    };

    let derived = ClassInfo {
        name: "Dog".to_string(),
        bases: vec!["Animal".to_string()],
        methods: vec![
            MethodInfo {
                name: "new".to_string(),
                is_abstract: false,
                params: vec!["name".to_string(), "breed".to_string()],
                return_type: None,
            },
            MethodInfo {
                name: "bark".to_string(),
                is_abstract: false,
                params: vec!["self".to_string()],
                return_type: Some("str".to_string()),
            },
        ],
        attributes: vec![
            AttributeInfo {
                name: "breed".to_string(),
                type_hint: Some("str".to_string()),
            },
        ],
        is_abstract: false,
    };

    translator.register_class(base);
    translator.register_class(derived);

    match translator.translate_class("Dog") {
        Ok(rust_code) => {
            println!("\nRust (using composition):");
            println!("{}", rust_code);

            println!("Note: super().__init__() becomes self.base = Animal::new(name)");
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!("\n{}\n", "=".repeat(80));
}

fn example_abstract_base_class() {
    println!("## Example 3: Abstract Base Class (ABC)\n");
    println!("Python:");
    println!(r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

    def describe(self) -> str:
        return f"Area: {{self.area()}}, Perimeter: {{self.perimeter()}}"

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius
"#);

    let mut translator = ClassInheritanceTranslator::new();

    let base = ClassInfo {
        name: "Shape".to_string(),
        bases: vec![],
        methods: vec![
            MethodInfo {
                name: "area".to_string(),
                is_abstract: true,
                params: vec!["self".to_string()],
                return_type: Some("float".to_string()),
            },
            MethodInfo {
                name: "perimeter".to_string(),
                is_abstract: true,
                params: vec!["self".to_string()],
                return_type: Some("float".to_string()),
            },
            MethodInfo {
                name: "describe".to_string(),
                is_abstract: false,
                params: vec!["self".to_string()],
                return_type: Some("str".to_string()),
            },
        ],
        attributes: vec![],
        is_abstract: true,
    };

    let derived = ClassInfo {
        name: "Circle".to_string(),
        bases: vec!["Shape".to_string()],
        methods: vec![
            MethodInfo {
                name: "new".to_string(),
                is_abstract: false,
                params: vec!["radius".to_string()],
                return_type: None,
            },
            MethodInfo {
                name: "area".to_string(),
                is_abstract: false,
                params: vec!["self".to_string()],
                return_type: Some("float".to_string()),
            },
            MethodInfo {
                name: "perimeter".to_string(),
                is_abstract: false,
                params: vec!["self".to_string()],
                return_type: Some("float".to_string()),
            },
        ],
        attributes: vec![
            AttributeInfo {
                name: "radius".to_string(),
                type_hint: Some("float".to_string()),
            },
        ],
        is_abstract: false,
    };

    translator.register_class(base);
    translator.register_class(derived);

    match translator.translate_class("Circle") {
        Ok(rust_code) => {
            println!("\nRust (using traits):");
            println!("{}", rust_code);

            println!("\nNote: Abstract methods become trait methods");
            println!("      Concrete methods become default implementations");
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!("\n{}\n", "=".repeat(80));
}

fn example_multiple_inheritance() {
    println!("## Example 4: Multiple Inheritance\n");
    println!("Python:");
    println!(r#"
class Flyable(ABC):
    @abstractmethod
    def fly(self) -> str:
        pass

class Swimmable(ABC):
    @abstractmethod
    def swim(self) -> str:
        pass

class Duck(Flyable, Swimmable):
    def __init__(self, name: str):
        self.name = name

    def fly(self) -> str:
        return f"{{self.name}} is flying"

    def swim(self) -> str:
        return f"{{self.name}} is swimming"
"#);

    let mut translator = ClassInheritanceTranslator::new();

    let flyable = ClassInfo {
        name: "Flyable".to_string(),
        bases: vec![],
        methods: vec![
            MethodInfo {
                name: "fly".to_string(),
                is_abstract: true,
                params: vec!["self".to_string()],
                return_type: Some("str".to_string()),
            },
        ],
        attributes: vec![],
        is_abstract: true,
    };

    let swimmable = ClassInfo {
        name: "Swimmable".to_string(),
        bases: vec![],
        methods: vec![
            MethodInfo {
                name: "swim".to_string(),
                is_abstract: true,
                params: vec!["self".to_string()],
                return_type: Some("str".to_string()),
            },
        ],
        attributes: vec![],
        is_abstract: true,
    };

    let duck = ClassInfo {
        name: "Duck".to_string(),
        bases: vec!["Flyable".to_string(), "Swimmable".to_string()],
        methods: vec![
            MethodInfo {
                name: "fly".to_string(),
                is_abstract: false,
                params: vec!["self".to_string()],
                return_type: Some("str".to_string()),
            },
            MethodInfo {
                name: "swim".to_string(),
                is_abstract: false,
                params: vec!["self".to_string()],
                return_type: Some("str".to_string()),
            },
        ],
        attributes: vec![
            AttributeInfo {
                name: "name".to_string(),
                type_hint: Some("str".to_string()),
            },
        ],
        is_abstract: false,
    };

    translator.register_class(flyable);
    translator.register_class(swimmable);
    translator.register_class(duck);

    match translator.translate_class("Duck") {
        Ok(rust_code) => {
            println!("\nRust (multiple trait implementations):");
            println!("{}", rust_code);

            println!("\nNote: Each Python base class becomes a Rust trait");
            println!("      The derived class implements all traits");
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!("\n{}\n", "=".repeat(80));
}

fn example_mro() {
    println!("## Example 5: Method Resolution Order (MRO)\n");
    println!("Python MRO follows C3 linearization:");
    println!("The order in which base classes are searched when looking for a method.\n");

    let mut hierarchy = ClassHierarchy::new();

    hierarchy.add_class(ClassInfo {
        name: "A".to_string(),
        bases: vec![],
        methods: vec![],
        attributes: vec![],
        is_abstract: false,
    });

    hierarchy.add_class(ClassInfo {
        name: "B".to_string(),
        bases: vec!["A".to_string()],
        methods: vec![],
        attributes: vec![],
        is_abstract: false,
    });

    hierarchy.add_class(ClassInfo {
        name: "C".to_string(),
        bases: vec!["A".to_string()],
        methods: vec![],
        attributes: vec![],
        is_abstract: false,
    });

    hierarchy.add_class(ClassInfo {
        name: "D".to_string(),
        bases: vec!["B".to_string(), "C".to_string()],
        methods: vec![],
        attributes: vec![],
        is_abstract: false,
    });

    let mro = hierarchy.compute_mro("D");

    println!("Class hierarchy:");
    println!("    A");
    println!("   / \\");
    println!("  B   C");
    println!("   \\ /");
    println!("    D");
    println!("\nMRO for class D: {:?}", mro);
    println!("\nIn Rust, this is handled by:");
    println!("1. Trait bounds to specify required capabilities");
    println!("2. Explicit method calls when ambiguous");
    println!("3. Type system ensures at compile time\n");

    println!("{}\n", "=".repeat(80));
}
