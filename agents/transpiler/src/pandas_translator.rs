//! Pandas to Polars Translation
//!
//! Translates Python Pandas DataFrame operations to Rust Polars,
//! supporting data manipulation, aggregation, filtering, and joins.

use std::collections::HashMap;

/// DataFrame creation method
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataFrameCreation {
    /// pd.DataFrame({...})
    FromDict,
    /// pd.DataFrame([...])
    FromList,
    /// pd.read_csv(path)
    ReadCsv,
    /// pd.read_json(path)
    ReadJson,
    /// pd.read_parquet(path)
    ReadParquet,
    /// pd.DataFrame(np.array(...))
    FromArray,
}

/// DataFrame operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataFrameOperation {
    /// Selection and filtering
    Select(SelectOp),
    /// Aggregation
    Aggregate(AggregateOp),
    /// Transformation
    Transform(TransformOp),
    /// Join/Merge
    Join(JoinOp),
    /// GroupBy
    GroupBy(GroupByOp),
    /// Reshaping
    Reshape(ReshapeOp),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectOp {
    /// df['col'] or df.col
    Column,
    /// df[['col1', 'col2']]
    Columns,
    /// df.loc[...]
    Loc,
    /// df.iloc[...]
    ILoc,
    /// df[df['col'] > 0]
    Filter,
    /// df.head(n)
    Head,
    /// df.tail(n)
    Tail,
    /// df.sample(n)
    Sample,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregateOp {
    Sum, Mean, Median, Std, Var, Min, Max, Count,
    Nunique, First, Last, Mode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformOp {
    /// df.apply(func)
    Apply,
    /// df.map(func)
    Map,
    /// df.assign(col=value)
    Assign,
    /// df.drop(cols)
    Drop,
    /// df.rename(cols)
    Rename,
    /// df.sort_values(by)
    Sort,
    /// df.drop_duplicates()
    DropDuplicates,
    /// df.fillna(value)
    FillNa,
    /// df.dropna()
    DropNa,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinOp {
    /// pd.merge(left, right, on='key')
    Merge,
    /// pd.concat([df1, df2])
    Concat,
    /// df.join(other)
    Join,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupByOp {
    /// df.groupby('col').agg(...)
    GroupBy,
    /// df.groupby('col').sum()
    GroupBySum,
    /// df.groupby('col').mean()
    GroupByMean,
    /// df.groupby('col').count()
    GroupByCount,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReshapeOp {
    /// df.pivot(...)
    Pivot,
    /// df.melt(...)
    Melt,
    /// df.stack()
    Stack,
    /// df.unstack()
    Unstack,
}

/// Pandas to Polars translator
pub struct PandasTranslator {
    /// Required imports
    imports: Vec<String>,
    /// DataFrame variable tracking
    dataframes: HashMap<String, String>,
}

impl PandasTranslator {
    pub fn new() -> Self {
        Self {
            imports: Vec::new(),
            dataframes: HashMap::new(),
        }
    }

    /// Translate DataFrame creation
    pub fn translate_creation(&mut self, method: &DataFrameCreation, args: &[String]) -> String {
        self.add_import("use polars::prelude::*");

        match method {
            DataFrameCreation::FromDict => {
                // pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
                // → df!["col1" => [1, 2], "col2" => [3, 4]]
                if args.is_empty() {
                    return "DataFrame::default()".to_string();
                }
                format!("df![{}]", args.join(", "))
            }
            DataFrameCreation::FromList => {
                // pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
                // → DataFrame::new(vec![...])
                format!("DataFrame::new(vec![{}]).unwrap()", args.join(", "))
            }
            DataFrameCreation::ReadCsv => {
                // pd.read_csv('data.csv')
                // → CsvReader::from_path("data.csv")?.finish()?
                self.add_import("use polars::prelude::CsvReader");
                if args.is_empty() {
                    return "CsvReader::from_path(\"data.csv\")?.finish()?".to_string();
                }
                format!("CsvReader::from_path({})?.finish()?", args[0])
            }
            DataFrameCreation::ReadJson => {
                // pd.read_json('data.json')
                // → JsonReader::from_path("data.json")?.finish()?
                self.add_import("use polars::prelude::JsonReader");
                if args.is_empty() {
                    return "JsonReader::from_path(\"data.json\")?.finish()?".to_string();
                }
                format!("JsonReader::from_path({})?.finish()?", args[0])
            }
            DataFrameCreation::ReadParquet => {
                // pd.read_parquet('data.parquet')
                // → ParquetReader::from_path("data.parquet")?.finish()?
                self.add_import("use polars::prelude::ParquetReader");
                if args.is_empty() {
                    return "ParquetReader::from_path(\"data.parquet\")?.finish()?".to_string();
                }
                format!("ParquetReader::from_path({})?.finish()?", args[0])
            }
            DataFrameCreation::FromArray => {
                // pd.DataFrame(np_array)
                // → DataFrame::new(vec![Series::new(...)])
                "/* DataFrame from array */".to_string()
            }
        }
    }

    /// Translate selection operation
    pub fn translate_select(&mut self, op: &SelectOp, df: &str, args: &[String]) -> String {
        match op {
            SelectOp::Column => {
                // df['col'] → df.column("col")?
                if args.is_empty() {
                    return format!("{}.column(\"col\")?", df);
                }
                format!("{}.column({})?", df, args[0])
            }
            SelectOp::Columns => {
                // df[['col1', 'col2']] → df.select([col("col1"), col("col2")])?
                if args.is_empty() {
                    return format!("{}.select([])?", df);
                }
                let cols: Vec<String> = args.iter()
                    .map(|c| format!("col({})", c))
                    .collect();
                format!("{}.select([{}])?", df, cols.join(", "))
            }
            SelectOp::Loc => {
                // df.loc[row, col] → df.select([col]).filter(...)
                format!("{}./* loc not direct equivalent */", df)
            }
            SelectOp::ILoc => {
                // df.iloc[0:5] → df.slice(0, 5)
                if args.len() >= 2 {
                    format!("{}.slice({}, {})", df, args[0], args[1])
                } else {
                    format!("{}.slice(0, 5)", df)
                }
            }
            SelectOp::Filter => {
                // df[df['col'] > 0] → df.filter(col("col").gt(0))?
                if args.is_empty() {
                    return format!("{}.filter(/* condition */)?", df);
                }
                format!("{}.filter({})?", df, args[0])
            }
            SelectOp::Head => {
                // df.head(10) → df.head(Some(10))
                if args.is_empty() {
                    format!("{}.head(Some(5))", df)
                } else {
                    format!("{}.head(Some({}))", df, args[0])
                }
            }
            SelectOp::Tail => {
                // df.tail(10) → df.tail(Some(10))
                if args.is_empty() {
                    format!("{}.tail(Some(5))", df)
                } else {
                    format!("{}.tail(Some({}))", df, args[0])
                }
            }
            SelectOp::Sample => {
                // df.sample(n) → df.sample_n(n, false, false, None)?
                if args.is_empty() {
                    format!("{}.sample_n(10, false, false, None)?", df)
                } else {
                    format!("{}.sample_n({}, false, false, None)?", df, args[0])
                }
            }
        }
    }

    /// Translate aggregation operation
    pub fn translate_aggregate(&mut self, op: &AggregateOp, df: &str, column: Option<&str>) -> String {
        let col_expr = if let Some(col) = column {
            format!("col(\"{}\")", col)
        } else {
            "col(\"*\")".to_string()
        };

        match op {
            AggregateOp::Sum => format!("{}.select([{}.sum()])?", df, col_expr),
            AggregateOp::Mean => format!("{}.select([{}.mean()])?", df, col_expr),
            AggregateOp::Median => format!("{}.select([{}.median()])?", df, col_expr),
            AggregateOp::Std => format!("{}.select([{}.std(1)])?", df, col_expr),
            AggregateOp::Var => format!("{}.select([{}.var(1)])?", df, col_expr),
            AggregateOp::Min => format!("{}.select([{}.min()])?", df, col_expr),
            AggregateOp::Max => format!("{}.select([{}.max()])?", df, col_expr),
            AggregateOp::Count => format!("{}.select([{}.count()])?", df, col_expr),
            AggregateOp::Nunique => format!("{}.select([{}.n_unique()])?", df, col_expr),
            AggregateOp::First => format!("{}.select([{}.first()])?", df, col_expr),
            AggregateOp::Last => format!("{}.select([{}.last()])?", df, col_expr),
            AggregateOp::Mode => format!("{}.select([{}.mode()])?", df, col_expr),
        }
    }

    /// Translate transformation operation
    pub fn translate_transform(&mut self, op: &TransformOp, df: &str, args: &[String]) -> String {
        match op {
            TransformOp::Apply => {
                // df.apply(func) → df.select([col("*").map(...)])
                format!("{}.select([col(\"*\").map(/* function */, ...)])?", df)
            }
            TransformOp::Map => {
                // Similar to apply
                format!("{}.select([col(\"*\").map(/* function */, ...)])?", df)
            }
            TransformOp::Assign => {
                // df.assign(new_col=value) → df.with_column(...)
                if args.len() >= 2 {
                    format!("{}.with_column({}.alias({}))?", df, args[1], args[0])
                } else {
                    format!("{}.with_column(/* expr */)?", df)
                }
            }
            TransformOp::Drop => {
                // df.drop(['col1', 'col2']) → df.drop(['col1', 'col2'])?
                if args.is_empty() {
                    return format!("{}.drop([])?", df);
                }
                format!("{}.drop([{}])?", df, args.join(", "))
            }
            TransformOp::Rename => {
                // df.rename({'old': 'new'}) → df.rename(['old'], ['new'])?
                format!("{}.rename([/* old */], [/* new */])?", df)
            }
            TransformOp::Sort => {
                // df.sort_values(by='col') → df.sort(['col'], false)?
                if args.is_empty() {
                    return format!("{}.sort([\"col\"], false)?", df);
                }
                format!("{}.sort([{}], false)?", df, args[0])
            }
            TransformOp::DropDuplicates => {
                // df.drop_duplicates() → df.unique(None, UniqueKeepStrategy::First)?
                format!("{}.unique(None, UniqueKeepStrategy::First)?", df)
            }
            TransformOp::FillNa => {
                // df.fillna(0) → df.fill_null(FillNullStrategy::Forward)?
                if args.is_empty() {
                    return format!("{}.fill_null(FillNullStrategy::Forward)?", df);
                }
                format!("{}.fill_null(FillNullStrategy::Value(AnyValue::from({})))?", df, args[0])
            }
            TransformOp::DropNa => {
                // df.dropna() → df.drop_nulls(None)?
                format!("{}.drop_nulls(None)?", df)
            }
        }
    }

    /// Translate join operation
    pub fn translate_join(&mut self, op: &JoinOp, args: &[String]) -> String {
        match op {
            JoinOp::Merge => {
                // pd.merge(left, right, on='key') → left.join(right, [col("key")], [col("key")], JoinType::Inner)?
                if args.len() >= 3 {
                    format!("{}.join({}, [col({})], [col({})], JoinType::Inner)?",
                        args[0], args[1], args[2], args[2])
                } else {
                    "/* merge */".to_string()
                }
            }
            JoinOp::Concat => {
                // pd.concat([df1, df2]) → concat([df1.lazy(), df2.lazy()], ...)?
                if args.is_empty() {
                    return "concat([/* dfs */], UnionArgs::default())?.collect()?".to_string();
                }
                format!("concat([{}], UnionArgs::default())?.collect()?", args.join(", "))
            }
            JoinOp::Join => {
                // df.join(other) → df.join(other, ...)?
                if !args.is_empty() {
                    format!("df.join({}, [col(\"index\")], [col(\"index\")], JoinType::Inner)?", args[0])
                } else {
                    "/* join */".to_string()
                }
            }
        }
    }

    /// Translate groupby operation
    pub fn translate_groupby(&mut self, op: &GroupByOp, df: &str, by: &[String], agg: Option<&str>) -> String {
        match op {
            GroupByOp::GroupBy => {
                // df.groupby('col').agg(...) → df.groupby(['col'])?.agg([...])?
                if by.is_empty() {
                    return format!("{}.groupby([\"col\"])?.agg([/* agg */])?", df);
                }
                let cols = by.join("\", \"");
                if let Some(agg_expr) = agg {
                    format!("{}.groupby([\"{}\"])?.agg([{}])?", df, cols, agg_expr)
                } else {
                    format!("{}.groupby([\"{}\"])?.agg([/* agg */])?", df, cols)
                }
            }
            GroupByOp::GroupBySum => {
                // df.groupby('col').sum() → df.groupby(['col'])?.sum()?
                if by.is_empty() {
                    return format!("{}.groupby([\"col\"])?.sum()?", df);
                }
                let cols = by.join("\", \"");
                format!("{}.groupby([\"{}\"])?.sum()?", df, cols)
            }
            GroupByOp::GroupByMean => {
                // df.groupby('col').mean() → df.groupby(['col'])?.mean()?
                if by.is_empty() {
                    return format!("{}.groupby([\"col\"])?.mean()?", df);
                }
                let cols = by.join("\", \"");
                format!("{}.groupby([\"{}\"])?.mean()?", df, cols)
            }
            GroupByOp::GroupByCount => {
                // df.groupby('col').count() → df.groupby(['col'])?.count()?
                if by.is_empty() {
                    return format!("{}.groupby([\"col\"])?.count()?", df);
                }
                let cols = by.join("\", \"");
                format!("{}.groupby([\"{}\"])?.count()?", df, cols)
            }
        }
    }

    /// Translate I/O operation
    pub fn translate_io(&mut self, operation: &str, df: &str, args: &[String]) -> String {
        match operation {
            "to_csv" => {
                // df.to_csv('output.csv') → CsvWriter::new(File::create("output.csv")?).finish(&mut df)?
                if args.is_empty() {
                    return format!("CsvWriter::new(File::create(\"output.csv\")?).finish(&mut {})?", df);
                }
                format!("CsvWriter::new(File::create({})?).finish(&mut {})?", args[0], df)
            }
            "to_json" => {
                // df.to_json('output.json') → JsonWriter::new(File::create("output.json")?).finish(&mut df)?
                if args.is_empty() {
                    return format!("JsonWriter::new(File::create(\"output.json\")?).finish(&mut {})?", df);
                }
                format!("JsonWriter::new(File::create({})?).finish(&mut {})?", args[0], df)
            }
            "to_parquet" => {
                // df.to_parquet('output.parquet') → ParquetWriter::new(File::create("output.parquet")?).finish(&mut df)?
                if args.is_empty() {
                    return format!("ParquetWriter::new(File::create(\"output.parquet\")?).finish(&mut {})?", df);
                }
                format!("ParquetWriter::new(File::create({})?).finish(&mut {})?", args[0], df)
            }
            _ => format!("/* {} */", operation),
        }
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

    /// Get Cargo dependencies
    pub fn get_cargo_dependencies(&self) -> Vec<(&str, &str)> {
        vec![
            ("polars", "0.35"),
        ]
    }
}

impl Default for PandasTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataframe_creation() {
        let mut translator = PandasTranslator::new();

        let result = translator.translate_creation(&DataFrameCreation::ReadCsv, &["\"data.csv\"".to_string()]);
        assert!(result.contains("CsvReader"));

        let result = translator.translate_creation(&DataFrameCreation::FromDict, &[]);
        assert!(result.contains("DataFrame") || result.contains("df!"));
    }

    #[test]
    fn test_selection() {
        let mut translator = PandasTranslator::new();

        let result = translator.translate_select(&SelectOp::Column, "df", &["\"col\"".to_string()]);
        assert!(result.contains("column"));

        let result = translator.translate_select(&SelectOp::Head, "df", &["10".to_string()]);
        assert!(result.contains("head"));
    }

    #[test]
    fn test_aggregation() {
        let mut translator = PandasTranslator::new();

        let result = translator.translate_aggregate(&AggregateOp::Sum, "df", Some("col"));
        assert!(result.contains("sum"));

        let result = translator.translate_aggregate(&AggregateOp::Mean, "df", Some("col"));
        assert!(result.contains("mean"));
    }

    #[test]
    fn test_groupby() {
        let mut translator = PandasTranslator::new();

        let result = translator.translate_groupby(&GroupByOp::GroupBySum, "df", &["col".to_string()], None);
        assert!(result.contains("groupby"));
        assert!(result.contains("sum"));
    }
}
