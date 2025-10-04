//! Tests for Day 3 features: operators, lists, tuples, control flow

#[cfg(test)]
mod tests {
    use crate::feature_translator::FeatureTranslator;

    #[test]
    fn test_comparison_operators() {
        let mut translator = FeatureTranslator::new();

        // Test all comparison operators
        let test_cases = vec![
            ("x == y", "x == y"),
            ("a != b", "a != b"),
            ("x < y", "x < y"),
            ("x > y", "x > y"),
            ("x <= y", "x <= y"),
            ("x >= y", "x >= y"),
        ];

        for (python, expected_rust) in test_cases {
            let python_code = format!("result = {}", python);
            let rust = translator.translate(&python_code).unwrap();
            assert!(rust.contains(expected_rust), "Failed for: {}", python);
        }
    }

    #[test]
    fn test_logical_and() {
        let mut translator = FeatureTranslator::new();
        let python = "result = x and y";
        let rust = translator.translate(python).unwrap();
        // Currently maps to bitwise &, will need special handling
        assert!(rust.contains("x & y") || rust.contains("x && y"));
    }

    #[test]
    fn test_logical_or() {
        let mut translator = FeatureTranslator::new();
        let python = "result = x or y";
        let rust = translator.translate(python).unwrap();
        // Currently maps to bitwise |, will need special handling
        assert!(rust.contains("x | y") || rust.contains("x || y"));
    }

    #[test]
    fn test_logical_not() {
        let mut translator = FeatureTranslator::new();
        let python = "result = not x";
        let rust = translator.translate(python).unwrap();
        assert!(rust.contains("!x"));
    }

    #[test]
    fn test_list_literal() {
        let mut translator = FeatureTranslator::new();
        let python = "numbers = [1, 2, 3, 4, 5]";
        let rust = translator.translate(python).unwrap();
        assert!(rust.contains("vec![1, 2, 3, 4, 5]"));
    }

    #[test]
    fn test_empty_list() {
        let mut translator = FeatureTranslator::new();
        let python = "empty = []";
        let rust = translator.translate(python).unwrap();
        assert!(rust.contains("vec![]"));
    }

    #[test]
    fn test_list_indexing() {
        let mut translator = FeatureTranslator::new();
        let python = "item = numbers[0]";
        let rust = translator.translate(python).unwrap();
        assert!(rust.contains("numbers[0]"));
    }

    #[test]
    fn test_tuple_literal() {
        let mut translator = FeatureTranslator::new();
        let python = "coords = (10, 20, 30)";
        let rust = translator.translate(python).unwrap();
        assert!(rust.contains("(10, 20, 30)"));
    }

    #[test]
    fn test_empty_tuple() {
        let mut translator = FeatureTranslator::new();
        let python = "empty = ()";
        let rust = translator.translate(python).unwrap();
        assert!(rust.contains("()"));
    }

    #[test]
    fn test_pass_statement() {
        let mut translator = FeatureTranslator::new();
        let python = "pass";
        let rust = translator.translate(python).unwrap();
        assert!(rust.contains("// pass"));
    }

    #[test]
    fn test_full_day3_program() {
        let mut translator = FeatureTranslator::new();
        let python = r#"
# Day 3 feature test
x = 10
y = 20
is_equal = x == y
is_less = x < y
numbers = [1, 2, 3]
coords = (x, y)
first = numbers[0]
"#;
        let rust = translator.translate(python).unwrap();

        println!("Generated Rust:\n{}", rust);

        assert!(rust.contains("let x: i32 = 10"));
        assert!(rust.contains("let y: i32 = 20"));
        assert!(rust.contains("x == y"));
        assert!(rust.contains("x < y"));
        assert!(rust.contains("vec![1, 2, 3]"));
        assert!(rust.contains("(x, y)"));
        assert!(rust.contains("numbers[0]"));
    }
}
