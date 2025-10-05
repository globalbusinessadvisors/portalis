# Migration Guide - Python to Rust

Comprehensive guide for migrating Python code to Rust using Portalis Transpiler.

---

## Table of Contents

1. [Core Language Features](#core-language-features)
2. [Data Structures](#data-structures)
3. [Object-Oriented Programming](#object-oriented-programming)
4. [Functional Programming](#functional-programming)
5. [Async/Concurrency](#asyncconcurrency)
6. [Error Handling](#error-handling)
7. [Standard Library](#standard-library)
8. [Third-Party Libraries](#third-party-libraries)
9. [Common Patterns](#common-patterns)
10. [Gotchas and Tips](#gotchas-and-tips)

---

## Core Language Features

### Variables and Types

#### Python
```python
# Dynamic typing
x = 42
x = "hello"  # Can change type

# Type hints (optional)
age: int = 25
name: str = "Alice"
```

#### Rust
```rust
// Static typing (required)
let x = 42;           // Type inferred as i32
// x = "hello";       // Error: type mismatch

// Explicit types
let age: i32 = 25;
let name: &str = "Alice";
```

**Key Differences**:
- Rust requires consistent types
- Type inference is strong but static
- Use `let mut` for mutable variables

#### Python
```python
count = 0
count += 1  # Can mutate by default
```

#### Rust
```rust
let mut count = 0;    // Must declare as mutable
count += 1;
```

---

### Functions

#### Python
```python
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str = "World") -> str:
    return f"Hello, {name}!"
```

#### Rust
```rust
fn add(a: i32, b: i32) -> i32 {
    a + b  // Implicit return (no semicolon)
}

fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

// Default arguments: use Option or separate function
fn greet_default() -> String {
    greet("World")
}
```

**Key Differences**:
- No default arguments in Rust (use Option<T> or builder pattern)
- Return type always required (or inferred as `()`)
- Implicit return without semicolon

---

### Control Flow

#### If/Else

**Python**:
```python
if x > 0:
    result = "positive"
elif x < 0:
    result = "negative"
else:
    result = "zero"
```

**Rust**:
```rust
let result = if x > 0 {
    "positive"
} else if x < 0 {
    "negative"
} else {
    "zero"
};
```

**Key Differences**:
- If is an expression in Rust (returns value)
- No parentheses around condition
- Braces required

#### Loops

**Python**:
```python
# For loop
for i in range(10):
    print(i)

# While loop
while count < 10:
    count += 1

# For-each
for item in items:
    process(item)
```

**Rust**:
```rust
// Range loop
for i in 0..10 {
    println!("{}", i);
}

// While loop
while count < 10 {
    count += 1;
}

// Iterator
for item in &items {
    process(item);
}
```

---

### Pattern Matching

#### Python (match - Python 3.10+)
```python
def describe(x):
    match x:
        case 0:
            return "zero"
        case 1 | 2:
            return "one or two"
        case _:
            return "other"
```

#### Rust
```rust
fn describe(x: i32) -> &'static str {
    match x {
        0 => "zero",
        1 | 2 => "one or two",
        _ => "other",
    }
}
```

**Enhanced in Rust**:
```rust
// Destructuring
match point {
    Point { x: 0, y: 0 } => "origin",
    Point { x: 0, y } => "y-axis",
    Point { x, y: 0 } => "x-axis",
    Point { x, y } => "somewhere",
}

// Guards
match number {
    n if n < 0 => "negative",
    0 => "zero",
    n if n > 0 => "positive",
}
```

---

## Data Structures

### Lists/Vectors

#### Python
```python
# List (dynamic array)
numbers = [1, 2, 3, 4, 5]
numbers.append(6)
numbers.extend([7, 8])

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in numbers if x % 2 == 0]
```

#### Rust
```rust
// Vec<T> (growable array)
let mut numbers = vec![1, 2, 3, 4, 5];
numbers.push(6);
numbers.extend([7, 8]);

// Iterator chains
let squares: Vec<i32> = (0..10).map(|x| x.pow(2)).collect();
let evens: Vec<i32> = numbers.iter().filter(|&x| x % 2 == 0).copied().collect();
```

**Key Differences**:
- Vec needs `mut` for modification
- Iterators are lazy (need `.collect()`)
- Type must be consistent

---

### Dictionaries/HashMaps

#### Python
```python
# Dictionary
user = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

# Access
name = user.get("name", "Unknown")

# Iteration
for key, value in user.items():
    print(f"{key}: {value}")
```

#### Rust
```rust
use std::collections::HashMap;

// HashMap
let mut user = HashMap::new();
user.insert("name", "Alice");
user.insert("age", "30");  // Note: values must be same type
user.insert("city", "NYC");

// Access
let name = user.get("name").unwrap_or(&"Unknown");

// Iteration
for (key, value) in &user {
    println!("{}: {}", key, value);
}
```

**Better: Use Struct**:
```rust
struct User {
    name: String,
    age: u32,
    city: String,
}

let user = User {
    name: "Alice".to_string(),
    age: 30,
    city: "NYC".to_string(),
};
```

---

### Tuples

#### Python
```python
point = (10, 20)
x, y = point

# Named tuple
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
```

#### Rust
```rust
let point = (10, 20);
let (x, y) = point;

// Better: Use struct
struct Point {
    x: i32,
    y: i32,
}

let p = Point { x: 10, y: 20 };
```

---

## Object-Oriented Programming

### Classes and Structs

#### Python
```python
class Rectangle:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def area(self) -> int:
        return self.width * self.height

    def perimeter(self) -> int:
        return 2 * (self.width + self.height)

    @property
    def is_square(self) -> bool:
        return self.width == self.height

    @staticmethod
    def create_square(size: int):
        return Rectangle(size, size)
```

#### Rust
```rust
struct Rectangle {
    width: i32,
    height: i32,
}

impl Rectangle {
    // Constructor (like __init__)
    fn new(width: i32, height: i32) -> Self {
        Rectangle { width, height }
    }

    // Instance method
    fn area(&self) -> i32 {
        self.width * self.height
    }

    fn perimeter(&self) -> i32 {
        2 * (self.width + self.height)
    }

    // Property-like (just a method)
    fn is_square(&self) -> bool {
        self.width == self.height
    }

    // Associated function (static method)
    fn create_square(size: i32) -> Self {
        Rectangle::new(size, size)
    }
}

// Usage
let rect = Rectangle::new(10, 20);
let area = rect.area();
let square = Rectangle::create_square(15);
```

---

### Inheritance

#### Python
```python
class Animal:
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return "Some sound"

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"
```

#### Rust (using Traits)
```rust
// Trait (like interface)
trait Animal {
    fn name(&self) -> &str;
    fn speak(&self) -> &str;
}

struct Dog {
    name: String,
}

struct Cat {
    name: String,
}

impl Animal for Dog {
    fn name(&self) -> &str {
        &self.name
    }

    fn speak(&self) -> &str {
        "Woof!"
    }
}

impl Animal for Cat {
    fn name(&self) -> &str {
        &self.name
    }

    fn speak(&self) -> &str {
        "Meow!"
    }
}

// Usage with polymorphism
fn make_animal_speak(animal: &dyn Animal) {
    println!("{} says {}", animal.name(), animal.speak());
}
```

**Key Differences**:
- Rust uses **composition over inheritance**
- Traits define behavior (like interfaces)
- Use `dyn Trait` for dynamic dispatch
- Use generics for static dispatch

---

## Functional Programming

### Lambda Functions

#### Python
```python
# Lambda
add = lambda x, y: x + y

# Map, filter, reduce
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

from functools import reduce
sum_all = reduce(lambda a, b: a + b, numbers)
```

#### Rust
```rust
// Closure
let add = |x, y| x + y;

// Iterator chains
let numbers = vec![1, 2, 3, 4, 5];
let doubled: Vec<i32> = numbers.iter().map(|x| x * 2).collect();
let evens: Vec<i32> = numbers.iter().filter(|&x| x % 2 == 0).copied().collect();

// Fold (like reduce)
let sum_all: i32 = numbers.iter().fold(0, |acc, &x| acc + x);
// Or simply:
let sum_all: i32 = numbers.iter().sum();
```

---

### Higher-Order Functions

#### Python
```python
def apply_twice(f, x):
    return f(f(x))

def add_one(x):
    return x + 1

result = apply_twice(add_one, 5)  # 7
```

#### Rust
```rust
fn apply_twice<F>(f: F, x: i32) -> i32
where
    F: Fn(i32) -> i32,
{
    f(f(x))
}

fn add_one(x: i32) -> i32 {
    x + 1
}

let result = apply_twice(add_one, 5);  // 7

// Or with closure
let result = apply_twice(|x| x + 1, 5);
```

---

### Generators/Iterators

#### Python
```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
for num in fibonacci(10):
    print(num)
```

#### Rust
```rust
fn fibonacci(n: usize) -> impl Iterator<Item = i32> {
    let mut a = 0;
    let mut b = 1;
    (0..n).map(move |_| {
        let current = a;
        let next = a + b;
        a = b;
        b = next;
        current
    })
}

// Usage
for num in fibonacci(10) {
    println!("{}", num);
}
```

---

## Async/Concurrency

### Async/Await

#### Python
```python
import asyncio

async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def main():
    urls = ["https://api1.com", "https://api2.com"]
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

asyncio.run(main())
```

#### Rust
```rust
use reqwest;
use serde_json::Value;

async fn fetch_data(url: &str) -> Result<Value, reqwest::Error> {
    reqwest::get(url).await?.json().await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let urls = vec!["https://api1.com", "https://api2.com"];

    let tasks: Vec<_> = urls
        .into_iter()
        .map(|url| fetch_data(url))
        .collect();

    let results = futures::future::join_all(tasks).await;
    Ok(())
}
```

**Key Differences**:
- Need async runtime (tokio, async-std)
- Results are `Result<T, E>` not exceptions
- Use `?` for error propagation

---

### Threading

#### Python
```python
from threading import Thread

def worker(n):
    for i in range(n):
        print(f"Worker {i}")

threads = []
for i in range(5):
    t = Thread(target=worker, args=(10,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

#### Rust
```rust
use std::thread;

fn worker(n: i32) {
    for i in 0..n {
        println!("Worker {}", i);
    }
}

fn main() {
    let mut handles = vec![];

    for _ in 0..5 {
        let handle = thread::spawn(|| worker(10));
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
```

**Better: Use Rayon for Data Parallelism**:
```rust
use rayon::prelude::*;

let results: Vec<i32> = (0..1000)
    .into_par_iter()
    .map(|x| expensive_computation(x))
    .collect();
```

---

## Error Handling

### Exceptions vs Result

#### Python
```python
def divide(a: int, b: int) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return 0.0
    except Exception as e:
        print(f"Error: {e}")
        raise

# Usage
result = divide(10, 2)
```

#### Rust
```rust
fn divide(a: i32, b: i32) -> Result<f64, String> {
    if b == 0 {
        Err("Cannot divide by zero".to_string())
    } else {
        Ok(a as f64 / b as f64)
    }
}

// Usage
match divide(10, 2) {
    Ok(result) => println!("Result: {}", result),
    Err(e) => println!("Error: {}", e),
}

// Or with ?
fn calculate() -> Result<f64, String> {
    let result = divide(10, 2)?;  // Propagates error
    Ok(result * 2.0)
}
```

**Key Differences**:
- Rust uses `Result<T, E>` type
- Errors are values, not exceptions
- Use `?` operator for error propagation
- Compiler enforces error handling

---

### Option vs None

#### Python
```python
def find_user(id: int) -> Optional[User]:
    if id in users:
        return users[id]
    return None

# Usage
user = find_user(123)
if user is not None:
    print(user.name)
```

#### Rust
```rust
fn find_user(id: i32) -> Option<User> {
    users.get(&id).cloned()
}

// Usage
match find_user(123) {
    Some(user) => println!("{}", user.name),
    None => println!("User not found"),
}

// Or
if let Some(user) = find_user(123) {
    println!("{}", user.name);
}

// Or with unwrap_or
let user = find_user(123).unwrap_or(default_user());
```

---

## Standard Library

### File I/O

#### Python
```python
# Read file
with open("data.txt", "r") as f:
    content = f.read()

# Write file
with open("output.txt", "w") as f:
    f.write("Hello, World!")

# Read lines
with open("data.txt", "r") as f:
    for line in f:
        print(line.strip())
```

#### Rust
```rust
use std::fs;
use std::io::{self, BufRead, Write};

// Read file
let content = fs::read_to_string("data.txt")?;

// Write file
fs::write("output.txt", "Hello, World!")?;

// Read lines
let file = fs::File::open("data.txt")?;
let reader = io::BufReader::new(file);
for line in reader.lines() {
    println!("{}", line?);
}
```

---

### JSON

#### Python
```python
import json

# Parse JSON
data = json.loads('{"name": "Alice", "age": 30}')

# Serialize
json_str = json.dumps(data)

# File I/O
with open("data.json", "r") as f:
    data = json.load(f)
```

#### Rust
```rust
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Serialize, Deserialize)]
struct Person {
    name: String,
    age: u32,
}

// Parse JSON
let data: Person = serde_json::from_str(r#"{"name": "Alice", "age": 30}"#)?;

// Serialize
let json_str = serde_json::to_string(&data)?;

// File I/O
let data: Person = serde_json::from_reader(fs::File::open("data.json")?)?;
```

---

## Third-Party Libraries

### HTTP Requests

**Python** (requests):
```python
import requests

response = requests.get("https://api.example.com/data")
data = response.json()

response = requests.post(
    "https://api.example.com/submit",
    json={"key": "value"}
)
```

**Rust** (reqwest):
```rust
use reqwest;

let response = reqwest::get("https://api.example.com/data")
    .await?
    .json::<serde_json::Value>()
    .await?;

let client = reqwest::Client::new();
let response = client
    .post("https://api.example.com/submit")
    .json(&serde_json::json!({"key": "value"}))
    .send()
    .await?;
```

---

### Data Processing

**Python** (pandas):
```python
import pandas as pd

df = pd.read_csv("data.csv")
filtered = df[df['age'] > 18]
grouped = df.groupby('category')['value'].sum()
```

**Rust** (polars):
```rust
use polars::prelude::*;

let df = CsvReader::from_path("data.csv")?.finish()?;

let filtered = df.filter(&df.column("age")?.gt(18)?)?;

let grouped = df
    .groupby(["category"])?
    .select(["value"])
    .sum()?;
```

---

## Common Patterns

### Builder Pattern

#### Python
```python
class Request:
    def __init__(self):
        self.url = None
        self.headers = {}
        self.timeout = 30

    def with_url(self, url):
        self.url = url
        return self

    def with_header(self, key, value):
        self.headers[key] = value
        return self

# Usage
request = Request()
    .with_url("https://api.example.com")
    .with_header("Authorization", "Bearer token")
```

#### Rust
```rust
struct Request {
    url: String,
    headers: HashMap<String, String>,
    timeout: u64,
}

impl Request {
    fn new() -> Self {
        Request {
            url: String::new(),
            headers: HashMap::new(),
            timeout: 30,
        }
    }

    fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

// Usage
let request = Request::new()
    .with_url("https://api.example.com")
    .with_header("Authorization", "Bearer token");
```

---

### Iterator Chaining

#### Python
```python
numbers = range(100)
result = [
    x * 2
    for x in numbers
    if x % 3 == 0 and x % 5 == 0
][:10]
```

#### Rust
```rust
let result: Vec<i32> = (0..100)
    .filter(|x| x % 3 == 0 && x % 5 == 0)
    .map(|x| x * 2)
    .take(10)
    .collect();
```

---

## Gotchas and Tips

### 1. Ownership and Borrowing

**Common Error**:
```rust
let s = String::from("hello");
let s2 = s;  // s is moved
// println!("{}", s);  // Error: s was moved
```

**Solution**:
```rust
let s = String::from("hello");
let s2 = s.clone();  // Clone if you need both
println!("{} {}", s, s2);

// Or borrow
let s = String::from("hello");
let s2 = &s;  // Borrow
println!("{} {}", s, s2);
```

### 2. String vs &str

**Python** has one string type, **Rust** has two:

```rust
let s1: String = String::from("hello");  // Owned, heap-allocated
let s2: &str = "hello";  // Borrowed, string slice

// Convert between them
let s3: String = s2.to_string();
let s4: &str = &s1;
```

**Tip**: Use `&str` for function parameters, `String` for owned data

### 3. Mutability

**Python**:
```python
x = 42
x = 43  # Always allowed
```

**Rust**:
```rust
let x = 42;
// x = 43;  // Error

let mut x = 42;
x = 43;  // OK
```

### 4. Integer Division

**Python**:
```python
print(5 / 2)   # 2.5 (float division)
print(5 // 2)  # 2 (integer division)
```

**Rust**:
```rust
println!("{}", 5 / 2);       // 2 (integer division)
println!("{}", 5.0 / 2.0);   // 2.5 (float division)
```

### 5. List Indices

**Python**:
```python
items = [1, 2, 3, 4, 5]
print(items[-1])  # 5 (negative indexing)
```

**Rust**:
```rust
let items = vec![1, 2, 3, 4, 5];
// No negative indexing
println!("{}", items[items.len() - 1]);  // 5
// Or
println!("{}", items.last().unwrap());
```

---

## Migration Checklist

- [ ] Identify Python libraries used → Find Rust equivalents
- [ ] Add type annotations to Python code (helps transpiler)
- [ ] Handle None/Optional explicitly
- [ ] Convert exceptions to Result types
- [ ] Review class hierarchies (may need trait redesign)
- [ ] Test async code (different runtime model)
- [ ] Check numeric types (i32 vs i64, overflow behavior)
- [ ] Verify string handling (&str vs String)
- [ ] Test error cases (Rust is stricter)
- [ ] Run tests after translation
- [ ] Optimize (dead code elimination, etc.)
- [ ] Build WASM if needed
- [ ] Profile and optimize hot paths

---

For more examples, see [EXAMPLES.md](./EXAMPLES.md).
For API details, see [API_REFERENCE.md](./API_REFERENCE.md).
For troubleshooting, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).
