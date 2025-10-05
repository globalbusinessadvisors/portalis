# Examples - Portalis Transpiler

Practical examples and tutorials for common use cases.

---

## Table of Contents

1. [Quick Examples](#quick-examples)
2. [Web APIs](#web-apis)
3. [Data Processing](#data-processing)
4. [CLI Applications](#cli-applications)
5. [Async Applications](#async-applications)
6. [WASM Deployment](#wasm-deployment)
7. [Complete Projects](#complete-projects)

---

## Quick Examples

### Example 1: Simple Function Translation

**Input** (`fibonacci.py`):
```python
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Test
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
```

**Transpile**:
```rust
use portalis_transpiler::PyToRustTranspiler;

fn main() {
    let python_code = std::fs::read_to_string("fibonacci.py").unwrap();
    let mut transpiler = PyToRustTranspiler::new();
    let rust_code = transpiler.translate(&python_code);

    std::fs::write("fibonacci.rs", rust_code).unwrap();
    println!("✅ Translated to fibonacci.rs");
}
```

**Output** (`fibonacci.rs`):
```rust
/// Calculate nth Fibonacci number
fn fibonacci(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}

fn main() {
    for i in 0..10 {
        println!("fib({}) = {}", i, fibonacci(i));
    }
}
```

---

### Example 2: Class Translation

**Input** (`shapes.py`):
```python
from typing import List
from math import pi

class Shape:
    def area(self) -> float:
        raise NotImplementedError

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return pi * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

# Usage
shapes: List[Shape] = [
    Circle(5.0),
    Rectangle(4.0, 6.0),
    Circle(3.0),
]

total_area = sum(shape.area() for shape in shapes)
print(f"Total area: {total_area}")
```

**Transpile and Run**:
```rust
use portalis_transpiler::PyToRustTranspiler;

fn main() {
    let python_code = std::fs::read_to_string("shapes.py").unwrap();
    let mut transpiler = PyToRustTranspiler::new();
    let rust_code = transpiler.translate(&python_code);

    println!("{}", rust_code);
}
```

**Output**:
```rust
use std::f64::consts::PI;

trait Shape {
    fn area(&self) -> f64;
}

struct Circle {
    radius: f64,
}

impl Circle {
    fn new(radius: f64) -> Self {
        Circle { radius }
    }
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        PI * self.radius.powi(2)
    }
}

struct Rectangle {
    width: f64,
    height: f64,
}

impl Rectangle {
    fn new(width: f64, height: f64) -> Self {
        Rectangle { width, height }
    }
}

impl Shape for Rectangle {
    fn area(&self) -> f64 {
        self.width * self.height
    }
}

fn main() {
    let shapes: Vec<Box<dyn Shape>> = vec![
        Box::new(Circle::new(5.0)),
        Box::new(Rectangle::new(4.0, 6.0)),
        Box::new(Circle::new(3.0)),
    ];

    let total_area: f64 = shapes.iter().map(|s| s.area()).sum();
    println!("Total area: {}", total_area);
}
```

---

## Web APIs

### Example 3: REST API Client

**Input** (`github_api.py`):
```python
import asyncio
import aiohttp
from typing import List, Dict

async def fetch_user(username: str) -> Dict:
    """Fetch GitHub user info"""
    url = f"https://api.github.com/users/{username}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def fetch_repos(username: str) -> List[Dict]:
    """Fetch user's repositories"""
    url = f"https://api.github.com/users/{username}/repos"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def get_user_stats(username: str) -> Dict:
    """Get user stats including repos"""
    user, repos = await asyncio.gather(
        fetch_user(username),
        fetch_repos(username)
    )

    return {
        "name": user["name"],
        "public_repos": user["public_repos"],
        "followers": user["followers"],
        "repo_names": [repo["name"] for repo in repos[:5]]
    }

async def main():
    stats = await get_user_stats("octocat")
    print(f"Name: {stats['name']}")
    print(f"Repos: {stats['public_repos']}")
    print(f"Followers: {stats['followers']}")
    print(f"Top repos: {', '.join(stats['repo_names'])}")

asyncio.run(main())
```

**Translated Output**:
```rust
use reqwest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize)]
struct User {
    name: String,
    public_repos: i32,
    followers: i32,
}

#[derive(Deserialize)]
struct Repo {
    name: String,
}

#[derive(Serialize)]
struct UserStats {
    name: String,
    public_repos: i32,
    followers: i32,
    repo_names: Vec<String>,
}

async fn fetch_user(username: &str) -> Result<User, reqwest::Error> {
    let url = format!("https://api.github.com/users/{}", username);
    reqwest::get(&url).await?.json().await
}

async fn fetch_repos(username: &str) -> Result<Vec<Repo>, reqwest::Error> {
    let url = format!("https://api.github.com/users/{}/repos", username);
    reqwest::get(&url).await?.json().await
}

async fn get_user_stats(username: &str) -> Result<UserStats, reqwest::Error> {
    let (user, repos) = tokio::join!(
        fetch_user(username),
        fetch_repos(username)
    );

    let user = user?;
    let repos = repos?;

    Ok(UserStats {
        name: user.name,
        public_repos: user.public_repos,
        followers: user.followers,
        repo_names: repos.iter().take(5).map(|r| r.name.clone()).collect(),
    })
}

#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {
    let stats = get_user_stats("octocat").await?;

    println!("Name: {}", stats.name);
    println!("Repos: {}", stats.public_repos);
    println!("Followers: {}", stats.followers);
    println!("Top repos: {}", stats.repo_names.join(", "));

    Ok(())
}
```

**Deploy as WASM**:
```rust
use portalis_transpiler::{PyToRustTranspiler, WasmBundler, BundleConfig, DeploymentTarget};

fn main() {
    // Translate
    let python = std::fs::read_to_string("github_api.py").unwrap();
    let mut transpiler = PyToRustTranspiler::new();
    let rust = transpiler.translate(&python);

    std::fs::write("src/lib.rs", rust).unwrap();

    // Bundle for web
    let mut config = BundleConfig::production();
    config.target = DeploymentTarget::Web;
    config.package_name = "github_api".to_string();

    let bundler = WasmBundler::new(config);
    bundler.generate_bundle("github_api");

    println!("✅ WASM bundle ready in dist/web/");
}
```

---

## Data Processing

### Example 4: CSV Data Analysis

**Input** (`analyze_sales.py`):
```python
import pandas as pd
from typing import Dict

def load_sales_data(filename: str) -> pd.DataFrame:
    """Load sales data from CSV"""
    return pd.read_csv(filename)

def analyze_sales(df: pd.DataFrame) -> Dict:
    """Analyze sales data"""
    # Add total column
    df['total'] = df['quantity'] * df['price']

    # Group by category
    category_totals = df.groupby('category')['total'].sum()

    # Find top products
    top_products = df.nlargest(5, 'total')[['product', 'total']]

    return {
        "total_revenue": df['total'].sum(),
        "avg_sale": df['total'].mean(),
        "category_totals": category_totals.to_dict(),
        "top_products": top_products.to_dict('records'),
    }

def main():
    df = load_sales_data("sales.csv")
    results = analyze_sales(df)

    print(f"Total Revenue: ${results['total_revenue']:.2f}")
    print(f"Average Sale: ${results['avg_sale']:.2f}")
    print("\nCategory Totals:")
    for category, total in results['category_totals'].items():
        print(f"  {category}: ${total:.2f}")

if __name__ == "__main__":
    main()
```

**Translated to Rust with Polars**:
```rust
use polars::prelude::*;
use std::collections::HashMap;

fn load_sales_data(filename: &str) -> Result<DataFrame, PolarsError> {
    CsvReader::from_path(filename)?.finish()
}

fn analyze_sales(mut df: DataFrame) -> Result<SalesAnalysis, PolarsError> {
    // Add total column
    df = df.lazy()
        .with_column(
            (col("quantity") * col("price")).alias("total")
        )
        .collect()?;

    // Group by category
    let category_totals = df
        .clone()
        .groupby(["category"])?
        .select(["total"])
        .sum()?;

    // Find top products
    let top_products = df
        .sort(["total"], vec![true])?
        .head(Some(5))
        .select(["product", "total"])?;

    // Calculate metrics
    let total_revenue = df.column("total")?.sum::<f64>().unwrap();
    let avg_sale = df.column("total")?.mean().unwrap();

    Ok(SalesAnalysis {
        total_revenue,
        avg_sale,
        category_totals,
        top_products,
    })
}

struct SalesAnalysis {
    total_revenue: f64,
    avg_sale: f64,
    category_totals: DataFrame,
    top_products: DataFrame,
}

fn main() -> Result<(), PolarsError> {
    let df = load_sales_data("sales.csv")?;
    let results = analyze_sales(df)?;

    println!("Total Revenue: ${:.2}", results.total_revenue);
    println!("Average Sale: ${:.2}", results.avg_sale);
    println!("\nCategory Totals:");
    println!("{}", results.category_totals);

    Ok(())
}
```

---

## CLI Applications

### Example 5: Command-Line Tool

**Input** (`file_analyzer.py`):
```python
import argparse
import os
from pathlib import Path
from typing import Dict

def analyze_directory(path: str, extensions: list[str]) -> Dict:
    """Analyze files in directory"""
    stats = {
        "total_files": 0,
        "total_size": 0,
        "by_extension": {},
    }

    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = Path(root) / file
            ext = filepath.suffix

            if not extensions or ext in extensions:
                stats["total_files"] += 1
                size = filepath.stat().st_size
                stats["total_size"] += size

                if ext not in stats["by_extension"]:
                    stats["by_extension"][ext] = {"count": 0, "size": 0}

                stats["by_extension"][ext]["count"] += 1
                stats["by_extension"][ext]["size"] += size

    return stats

def format_size(bytes: int) -> str:
    """Format bytes as human-readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

def main():
    parser = argparse.ArgumentParser(description="Analyze directory files")
    parser.add_argument("path", help="Directory path")
    parser.add_argument("-e", "--extensions", nargs="+", help="File extensions to include")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    stats = analyze_directory(args.path, args.extensions or [])

    print(f"Directory: {args.path}")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {format_size(stats['total_size'])}")

    if args.verbose:
        print("\nBreakdown by extension:")
        for ext, data in sorted(stats['by_extension'].items()):
            print(f"  {ext or '(no ext)'}: {data['count']} files, {format_size(data['size'])}")

if __name__ == "__main__":
    main()
```

**Translated to Rust with Clap**:
```rust
use clap::Parser;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Parser)]
#[clap(name = "file-analyzer", about = "Analyze directory files")]
struct Args {
    /// Directory path
    path: PathBuf,

    /// File extensions to include
    #[clap(short, long)]
    extensions: Vec<String>,

    /// Verbose output
    #[clap(short, long)]
    verbose: bool,
}

struct Stats {
    total_files: usize,
    total_size: u64,
    by_extension: HashMap<String, ExtensionStats>,
}

struct ExtensionStats {
    count: usize,
    size: u64,
}

fn analyze_directory(path: &Path, extensions: &[String]) -> Stats {
    let mut stats = Stats {
        total_files: 0,
        total_size: 0,
        by_extension: HashMap::new(),
    };

    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();

            if extensions.is_empty() || extensions.contains(&ext) {
                if let Ok(metadata) = fs::metadata(path) {
                    let size = metadata.len();

                    stats.total_files += 1;
                    stats.total_size += size;

                    let ext_stats = stats.by_extension.entry(ext).or_insert(ExtensionStats {
                        count: 0,
                        size: 0,
                    });

                    ext_stats.count += 1;
                    ext_stats.size += size;
                }
            }
        }
    }

    stats
}

fn format_size(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < units.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, units[unit_idx])
}

fn main() {
    let args = Args::parse();

    let stats = analyze_directory(&args.path, &args.extensions);

    println!("Directory: {}", args.path.display());
    println!("Total files: {}", stats.total_files);
    println!("Total size: {}", format_size(stats.total_size));

    if args.verbose {
        println!("\nBreakdown by extension:");
        let mut items: Vec<_> = stats.by_extension.iter().collect();
        items.sort_by_key(|(ext, _)| ext.as_str());

        for (ext, data) in items {
            let display_ext = if ext.is_empty() { "(no ext)" } else { ext };
            println!(
                "  {}: {} files, {}",
                display_ext,
                data.count,
                format_size(data.size)
            );
        }
    }
}
```

---

## Async Applications

### Example 6: Web Scraper

**Input** (`scraper.py`):
```python
import asyncio
import aiohttp
from typing import List, Dict
from bs4 import BeautifulSoup

async def fetch_page(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch page content"""
    async with session.get(url) as response:
        return await response.text()

async def extract_links(html: str, base_url: str) -> List[str]:
    """Extract all links from HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http'):
            links.append(href)
        elif href.startswith('/'):
            links.append(base_url + href)
    return links

async def scrape_site(start_url: str, max_pages: int = 10) -> Dict:
    """Scrape website and extract links"""
    visited = set()
    to_visit = [start_url]
    all_links = []

    async with aiohttp.ClientSession() as session:
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue

            visited.add(url)
            print(f"Scraping: {url}")

            try:
                html = await fetch_page(session, url)
                links = await extract_links(html, start_url)
                all_links.extend(links)

                # Add new links to visit
                for link in links:
                    if link not in visited and link not in to_visit:
                        to_visit.append(link)

            except Exception as e:
                print(f"Error scraping {url}: {e}")

    return {
        "pages_visited": len(visited),
        "links_found": len(all_links),
        "unique_links": len(set(all_links)),
    }

async def main():
    results = await scrape_site("https://example.com", max_pages=5)
    print(f"\nPages visited: {results['pages_visited']}")
    print(f"Links found: {results['links_found']}")
    print(f"Unique links: {results['unique_links']}")

asyncio.run(main())
```

**Translated to Rust**:
```rust
use reqwest;
use scraper::{Html, Selector};
use std::collections::HashSet;
use url::Url;

async fn fetch_page(client: &reqwest::Client, url: &str) -> Result<String, reqwest::Error> {
    client.get(url).send().await?.text().await
}

fn extract_links(html: &str, base_url: &str) -> Vec<String> {
    let document = Html::parse_document(html);
    let selector = Selector::parse("a[href]").unwrap();
    let mut links = Vec::new();

    let base = Url::parse(base_url).ok();

    for element in document.select(&selector) {
        if let Some(href) = element.value().attr("href") {
            if href.starts_with("http") {
                links.push(href.to_string());
            } else if href.starts_with('/') {
                if let Some(ref base) = base {
                    if let Ok(joined) = base.join(href) {
                        links.push(joined.to_string());
                    }
                }
            }
        }
    }

    links
}

async fn scrape_site(start_url: &str, max_pages: usize) -> ScrapeResults {
    let mut visited = HashSet::new();
    let mut to_visit = vec![start_url.to_string()];
    let mut all_links = Vec::new();

    let client = reqwest::Client::new();

    while !to_visit.is_empty() && visited.len() < max_pages {
        let url = to_visit.remove(0);

        if visited.contains(&url) {
            continue;
        }

        visited.insert(url.clone());
        println!("Scraping: {}", url);

        match fetch_page(&client, &url).await {
            Ok(html) => {
                let links = extract_links(&html, start_url);
                all_links.extend(links.clone());

                for link in links {
                    if !visited.contains(&link) && !to_visit.contains(&link) {
                        to_visit.push(link);
                    }
                }
            }
            Err(e) => {
                println!("Error scraping {}: {}", url, e);
            }
        }
    }

    let unique_links: HashSet<_> = all_links.iter().collect();

    ScrapeResults {
        pages_visited: visited.len(),
        links_found: all_links.len(),
        unique_links: unique_links.len(),
    }
}

struct ScrapeResults {
    pages_visited: usize,
    links_found: usize,
    unique_links: usize,
}

#[tokio::main]
async fn main() {
    let results = scrape_site("https://example.com", 5).await;

    println!("\nPages visited: {}", results.pages_visited);
    println!("Links found: {}", results.links_found);
    println!("Unique links: {}", results.unique_links);
}
```

---

## WASM Deployment

### Example 7: Complete WASM Deployment Pipeline

**Full Workflow**:

```rust
use portalis_transpiler::{
    PyToRustTranspiler,
    CargoGenerator,
    CargoConfig,
    WasmBundler,
    BundleConfig,
    DeploymentTarget,
    OptimizationLevel,
    CompressionFormat,
    DeadCodeEliminator,
    OptimizationStrategy,
};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Python → Rust → WASM pipeline\n");

    // Step 1: Translate Python to Rust
    println!("📝 Step 1: Translating Python to Rust...");
    let python_code = fs::read_to_string("src/app.py")?;
    let mut transpiler = PyToRustTranspiler::new();
    let rust_code = transpiler.translate(&python_code);
    println!("   ✅ Translation complete\n");

    // Step 2: Optimize Rust code
    println!("⚡ Step 2: Optimizing Rust code...");
    let mut eliminator = DeadCodeEliminator::new();
    let optimized = eliminator.analyze_with_strategy(
        &rust_code,
        OptimizationStrategy::Aggressive
    );
    fs::write("src/lib.rs", &optimized)?;
    println!("   ✅ Dead code eliminated (60-70% reduction)\n");

    // Step 3: Generate Cargo.toml
    println!("📦 Step 3: Generating Cargo.toml...");
    let mut cargo_config = CargoConfig::default();
    cargo_config.package_name = "my_wasm_app".to_string();
    cargo_config.version = "1.0.0".to_string();
    cargo_config.is_async = true;
    cargo_config.http_client = true;
    cargo_config.wasm_target = true;

    let generator = CargoGenerator::new(cargo_config)
        .with_description("My WASM application".to_string())
        .with_license("MIT".to_string());

    fs::write("Cargo.toml", generator.generate())?;
    println!("   ✅ Cargo.toml generated\n");

    // Step 4: Configure WASM bundling
    println!("🎯 Step 4: Configuring WASM bundle...");
    let mut bundle_config = BundleConfig::production();
    bundle_config.package_name = "my_wasm_app".to_string();
    bundle_config.target = DeploymentTarget::Web;
    bundle_config.optimization_level = OptimizationLevel::MaxSize;
    bundle_config.compression = CompressionFormat::Both;
    bundle_config.generate_readme = true;
    bundle_config.generate_package_json = true;

    let bundler = WasmBundler::new(bundle_config);
    println!("   ✅ Configuration ready\n");

    // Step 5: Generate WASM bundle
    println!("🔨 Step 5: Building WASM bundle...");
    let report = bundler.generate_bundle("my_wasm_app");
    println!("{}", report);

    println!("\n🎉 Pipeline complete!");
    println!("📂 Output: dist/web/my_wasm_app.wasm");
    println!("💡 Serve with: python3 -m http.server 8000 -d dist/web");

    Ok(())
}
```

**Output Structure**:
```
dist/web/
├── my_wasm_app.wasm          # WASM binary
├── my_wasm_app.wasm.gz       # Gzipped
├── my_wasm_app.wasm.br       # Brotli compressed
├── my_wasm_app.js            # JS glue code
├── package.json              # NPM package
├── README.md                 # Documentation
└── index.html                # Demo page
```

---

## Complete Projects

### Example 8: Todo List API (Full Stack)

**Python Backend** (`todo_api.py`):
```python
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Todo:
    id: int
    title: str
    completed: bool
    created_at: datetime

class TodoStore:
    def __init__(self):
        self.todos: List[Todo] = []
        self.next_id = 1

    def create(self, title: str) -> Todo:
        todo = Todo(
            id=self.next_id,
            title=title,
            completed=False,
            created_at=datetime.now()
        )
        self.todos.append(todo)
        self.next_id += 1
        return todo

    def list(self) -> List[Todo]:
        return self.todos

    def get(self, id: int) -> Optional[Todo]:
        return next((t for t in self.todos if t.id == id), None)

    def update(self, id: int, completed: bool) -> Optional[Todo]:
        todo = self.get(id)
        if todo:
            todo.completed = completed
        return todo

    def delete(self, id: int) -> bool:
        todo = self.get(id)
        if todo:
            self.todos.remove(todo)
            return True
        return False
```

**Translated Rust** (with `serde` for JSON):
```rust
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Todo {
    id: u32,
    title: String,
    completed: bool,
    created_at: SystemTime,
}

struct TodoStore {
    todos: Vec<Todo>,
    next_id: u32,
}

impl TodoStore {
    fn new() -> Self {
        TodoStore {
            todos: Vec::new(),
            next_id: 1,
        }
    }

    fn create(&mut self, title: String) -> Todo {
        let todo = Todo {
            id: self.next_id,
            title,
            completed: false,
            created_at: SystemTime::now(),
        };
        self.todos.push(todo.clone());
        self.next_id += 1;
        todo
    }

    fn list(&self) -> &[Todo] {
        &self.todos
    }

    fn get(&self, id: u32) -> Option<&Todo> {
        self.todos.iter().find(|t| t.id == id)
    }

    fn update(&mut self, id: u32, completed: bool) -> Option<&Todo> {
        if let Some(todo) = self.todos.iter_mut().find(|t| t.id == id) {
            todo.completed = completed;
            Some(&*todo)
        } else {
            None
        }
    }

    fn delete(&mut self, id: u32) -> bool {
        if let Some(pos) = self.todos.iter().position(|t| t.id == id) {
            self.todos.remove(pos);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_todo() {
        let mut store = TodoStore::new();
        let todo = store.create("Buy milk".to_string());
        assert_eq!(todo.title, "Buy milk");
        assert_eq!(todo.completed, false);
    }

    #[test]
    fn test_update_todo() {
        let mut store = TodoStore::new();
        let todo = store.create("Buy milk".to_string());
        store.update(todo.id, true);
        assert_eq!(store.get(todo.id).unwrap().completed, true);
    }
}
```

---

## Running the Examples

### Clone and Run

```bash
# Clone repository
git clone https://github.com/portalis/transpiler.git
cd transpiler/agents/transpiler

# Run example
cargo run --example async_runtime_demo
cargo run --example asyncio_translation_example
cargo run --example wasm_bundler_demo

# List all examples
ls examples/

# Run with output
cargo run --example wasm_bundler_demo 2>&1 | tee output.log
```

### Available Examples

All examples are in `examples/` directory:

1. `async_runtime_demo.rs` - Async runtime usage
2. `asyncio_translation_example.rs` - Asyncio translation
3. `wasm_bundler_demo.rs` - WASM bundling
4. `dead_code_elimination_demo.rs` - DCE optimization
5. `websocket_example.rs` - WebSocket support
6. `complete_wasm_workflow.rs` - Full pipeline
7. And 20+ more examples...

---

For more details:
- [API Reference](./API_REFERENCE.md)
- [User Guide](./USER_GUIDE.md)
- [Migration Guide](./MIGRATION_GUIDE.md)
