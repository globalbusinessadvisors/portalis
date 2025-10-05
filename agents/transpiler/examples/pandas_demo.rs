//! Pandas to Polars Translation Examples
//!
//! Demonstrates translation of Python Pandas DataFrame operations to Rust Polars,
//! covering data manipulation, filtering, aggregation, and joins.

use portalis_transpiler::pandas_translator::*;

fn main() {
    println!("=== Pandas to Polars Translation Examples ===\n");

    // Example 1: DataFrame creation
    example_dataframe_creation();

    // Example 2: Selection and filtering
    example_selection_filtering();

    // Example 3: Aggregation operations
    example_aggregations();

    // Example 4: Data transformations
    example_transformations();

    // Example 5: GroupBy operations
    example_groupby();

    // Example 6: Joins and merges
    example_joins();

    // Example 7: I/O operations
    example_io_operations();

    // Example 8: Real-world data analysis
    example_data_analysis();
}

fn example_dataframe_creation() {
    println!("## Example 1: DataFrame Creation\n");

    let mut translator = PandasTranslator::new();

    println!("Python Pandas                           →  Rust Polars");
    println!("{}", "-".repeat(80));

    let examples = vec![
        (DataFrameCreation::FromDict, vec![], "Create from dict",
         "pd.DataFrame({{'col1': [1, 2], 'col2': [3, 4]}}"),
        (DataFrameCreation::ReadCsv, vec!["\"data.csv\"".to_string()], "Read CSV",
         "pd.read_csv('data.csv')"),
        (DataFrameCreation::ReadJson, vec!["\"data.json\"".to_string()], "Read JSON",
         "pd.read_json('data.json')"),
        (DataFrameCreation::ReadParquet, vec!["\"data.parquet\"".to_string()], "Read Parquet",
         "pd.read_parquet('data.parquet')"),
    ];

    for (method, args, desc, python) in examples {
        let rust = translator.translate_creation(&method, &args);
        println!("{:<40} # {}", python, desc);
        println!("{:>40}   → {}", "", rust);
        println!();
    }

    println!("\n{}\n", "=".repeat(80));
}

fn example_selection_filtering() {
    println!("## Example 2: Selection and Filtering\n");

    let mut translator = PandasTranslator::new();

    println!("Python Pandas                →  Rust Polars");
    println!("{}", "-".repeat(70));

    let selections = vec![
        (SelectOp::Column, "df", vec!["\"price\"".to_string()], "df['price']", "Select column"),
        (SelectOp::Columns, "df", vec!["\"price\"".to_string(), "\"qty\"".to_string()], "df[['price', 'qty']]", "Select columns"),
        (SelectOp::Head, "df", vec!["10".to_string()], "df.head(10)", "First 10 rows"),
        (SelectOp::Tail, "df", vec!["5".to_string()], "df.tail(5)", "Last 5 rows"),
        (SelectOp::Filter, "df", vec!["col(\"price\").gt(100)".to_string()], "df[df['price'] > 100]", "Filter rows"),
        (SelectOp::Sample, "df", vec!["10".to_string()], "df.sample(10)", "Random 10 rows"),
    ];

    for (op, df, args, python, desc) in selections {
        let rust = translator.translate_select(&op, df, &args);
        println!("{:<30} # {}", python, desc);
        println!("{:>32}  → {}", "", rust);
        println!();
    }

    println!("Boolean filtering:");
    println!("  df[df['age'] > 18]        →  df.filter(col(\"age\").gt(18))?");
    println!("  df[df['name'] == 'Alice'] →  df.filter(col(\"name\").eq(\"Alice\"))?");
    println!("  df[(df['a'] > 0) & ...]   →  df.filter(col(\"a\").gt(0).and(...))?");

    println!("\n{}\n", "=".repeat(80));
}

fn example_aggregations() {
    println!("## Example 3: Aggregation Operations\n");

    let mut translator = PandasTranslator::new();

    println!("Python Pandas            →  Rust Polars");
    println!("{}", "-".repeat(70));

    let aggs = vec![
        (AggregateOp::Sum, "df['price'].sum()", "Sum column"),
        (AggregateOp::Mean, "df['price'].mean()", "Mean"),
        (AggregateOp::Median, "df['price'].median()", "Median"),
        (AggregateOp::Std, "df['price'].std()", "Std deviation"),
        (AggregateOp::Min, "df['price'].min()", "Minimum"),
        (AggregateOp::Max, "df['price'].max()", "Maximum"),
        (AggregateOp::Count, "df['price'].count()", "Count"),
        (AggregateOp::Nunique, "df['category'].nunique()", "Unique count"),
    ];

    for (op, python, desc) in aggs {
        let rust = translator.translate_aggregate(&op, "df", Some("price"));
        println!("{:<30} # {}", python, desc);
        println!("{:>32}  → {}", "", rust);
        println!();
    }

    println!("Multiple aggregations:");
    println!("  df.agg({{'price': 'sum', 'qty': 'mean'}})");
    println!("  → df.select([col(\"price\").sum(), col(\"qty\").mean()])?");

    println!("\n{}\n", "=".repeat(80));
}

fn example_transformations() {
    println!("## Example 4: Data Transformations\n");

    let mut translator = PandasTranslator::new();

    println!("Python Pandas                  →  Rust Polars");
    println!("{}", "-".repeat(70));

    let transforms = vec![
        (TransformOp::Assign, vec!["\"total\"".to_string(), "col(\"price\") * col(\"qty\")".to_string()],
         "df.assign(total=df['price'] * df['qty'])", "Add column"),
        (TransformOp::Drop, vec!["\"col1\"".to_string()],
         "df.drop(['col1'])", "Drop column"),
        (TransformOp::Sort, vec!["\"price\"".to_string()],
         "df.sort_values('price')", "Sort"),
        (TransformOp::DropDuplicates, vec![],
         "df.drop_duplicates()", "Remove duplicates"),
        (TransformOp::FillNa, vec!["0".to_string()],
         "df.fillna(0)", "Fill nulls"),
        (TransformOp::DropNa, vec![],
         "df.dropna()", "Drop nulls"),
    ];

    for (op, args, python, desc) in transforms {
        let rust = translator.translate_transform(&op, "df", &args);
        println!("{:<40} # {}", python, desc);
        println!("{:>42}  → {}", "", rust);
        println!();
    }

    println!("Column operations:");
    println!("  df['new'] = df['a'] + df['b']  →  df.with_column((col(\"a\") + col(\"b\")).alias(\"new\"))?");
    println!("  df.rename({{'old': 'new'}})     →  df.rename([\"old\"], [\"new\"])?");

    println!("\n{}\n", "=".repeat(80));
}

fn example_groupby() {
    println!("## Example 5: GroupBy Operations\n");

    let mut translator = PandasTranslator::new();

    println!("Python Pandas                         →  Rust Polars");
    println!("{}", "-".repeat(75));

    let groupby_ops = vec![
        (GroupByOp::GroupBySum, vec!["category".to_string()], None,
         "df.groupby('category').sum()", "Sum by group"),
        (GroupByOp::GroupByMean, vec!["category".to_string()], None,
         "df.groupby('category').mean()", "Mean by group"),
        (GroupByOp::GroupByCount, vec!["category".to_string()], None,
         "df.groupby('category').count()", "Count by group"),
    ];

    for (op, by, agg, python, desc) in groupby_ops {
        let rust = translator.translate_groupby(&op, "df", &by, agg);
        println!("{:<45} # {}", python, desc);
        println!("{:>47}  → {}", "", rust);
        println!();
    }

    println!("Custom aggregations:");
    println!("  df.groupby('category').agg({{'price': 'sum', 'qty': 'mean'}})");
    println!("  → df.groupby([\"category\"])?.agg([col(\"price\").sum(), col(\"qty\").mean()])?");
    println!();

    println!("Multiple groupby columns:");
    println!("  df.groupby(['category', 'region']).sum()");
    println!("  → df.groupby([\"category\", \"region\"])?.sum()?");

    println!("\n{}\n", "=".repeat(80));
}

fn example_joins() {
    println!("## Example 6: Joins and Merges\n");

    let mut translator = PandasTranslator::new();

    println!("Python Pandas                                →  Rust Polars");
    println!("{}", "-".repeat(75));

    println!("Inner join:");
    println!("  pd.merge(df1, df2, on='id')");
    let rust = translator.translate_join(&JoinOp::Merge,
        &["df1".to_string(), "df2".to_string(), "\"id\"".to_string()]);
    println!("  → {}\n", rust);

    println!("Concatenation:");
    println!("  pd.concat([df1, df2])");
    let rust = translator.translate_join(&JoinOp::Concat,
        &["df1.lazy()".to_string(), "df2.lazy()".to_string()]);
    println!("  → {}\n", rust);

    println!("Join types:");
    println!("  Inner: JoinType::Inner");
    println!("  Left:  JoinType::Left");
    println!("  Right: JoinType::Right  ");
    println!("  Outer: JoinType::Outer");
    println!("  Cross: JoinType::Cross");

    println!("\n{}\n", "=".repeat(80));
}

fn example_io_operations() {
    println!("## Example 7: I/O Operations\n");

    let mut translator = PandasTranslator::new();

    println!("Python Pandas                →  Rust Polars");
    println!("{}", "-".repeat(70));

    println!("Writing data:");
    println!("  df.to_csv('output.csv')");
    let rust = translator.translate_io("to_csv", "df", &["\"output.csv\"".to_string()]);
    println!("  → {}\n", rust);

    println!("  df.to_json('output.json')");
    let rust = translator.translate_io("to_json", "df", &["\"output.json\"".to_string()]);
    println!("  → {}\n", rust);

    println!("  df.to_parquet('output.parquet')");
    let rust = translator.translate_io("to_parquet", "df", &["\"output.parquet\"".to_string()]);
    println!("  → {}\n", rust);

    println!("Reading data (from Example 1):");
    println!("  pd.read_csv('data.csv')    →  CsvReader::from_path(\"data.csv\")?.finish()?");
    println!("  pd.read_json('data.json')  →  JsonReader::from_path(\"data.json\")?.finish()?");

    println!("\n{}\n", "=".repeat(80));
}

fn example_data_analysis() {
    println!("## Example 8: Real-world Data Analysis\n");

    let mut translator = PandasTranslator::new();

    println!("Example: Sales data analysis\n");

    println!("Python Pandas:");
    println!(r#"
import pandas as pd

# Load data
df = pd.read_csv('sales.csv')

# Filter recent sales
recent = df[df['date'] >= '2024-01-01']

# Calculate total by category
category_sales = recent.groupby('category').agg({{
    'amount': 'sum',
    'quantity': 'sum',
    'order_id': 'count'
}}).rename(columns={{'order_id': 'num_orders'}})

# Sort by total amount
top_categories = category_sales.sort_values('amount', ascending=False).head(10)

# Save results
top_categories.to_csv('top_categories.csv')
"#);

    println!("\nRust Polars:");
    println!(r#"
use polars::prelude::*;

// Load data
let df = CsvReader::from_path("sales.csv")?.finish()?;

// Filter recent sales
let recent = df.filter(
    col("date").gt_eq(lit("2024-01-01"))
)?;

// Calculate total by category
let category_sales = recent
    .groupby(["category"])?
    .agg([
        col("amount").sum().alias("total_amount"),
        col("quantity").sum().alias("total_quantity"),
        col("order_id").count().alias("num_orders"),
    ])?;

// Sort by total amount
let top_categories = category_sales
    .sort(["total_amount"], true)?  // true = descending
    .head(Some(10));

// Save results
CsvWriter::new(File::create("top_categories.csv")?)
    .finish(&mut top_categories)?;
"#);

    println!("\nKey translation points:");
    println!("  1. Read CSV: pd.read_csv → CsvReader::from_path().finish()");
    println!("  2. Filter: df[condition] → df.filter(col(...).gt_eq(...))");
    println!("  3. GroupBy agg: .agg(dict) → .agg([col().sum().alias(...)])");
    println!("  4. Sort: .sort_values → .sort([...], descending)");
    println!("  5. Head: .head(n) → .head(Some(n))");
    println!("  6. Write CSV: .to_csv → CsvWriter::new().finish()");

    println!("\nPerformance benefits:");
    println!("  ✓ Polars is written in Rust - much faster than Pandas");
    println!("  ✓ Lazy evaluation - optimizes query plan");
    println!("  ✓ Parallel execution - uses all CPU cores");
    println!("  ✓ Memory efficient - Apache Arrow backend");

    println!("\nRequired dependency:");
    for (name, version) in translator.get_cargo_dependencies() {
        println!("  {} = \"{}\"", name, version);
    }

    println!("\n{}\n", "=".repeat(80));
}
