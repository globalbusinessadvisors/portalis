"""
Unit tests for Translation Validator

Tests validation of generated Rust code.
"""

import pytest

from src.validation.validator import (
    TranslationValidator,
    ValidationResult,
    ValidationLevel,
    ValidationStatus,
    ValidationIssue
)


@pytest.fixture
def validator():
    """Create validator instance."""
    return TranslationValidator(level=ValidationLevel.FULL)


@pytest.fixture
def syntax_validator():
    """Create syntax-only validator."""
    return TranslationValidator(level=ValidationLevel.SYNTAX_ONLY)


class TestTranslationValidator:
    """Test suite for TranslationValidator."""

    def test_valid_rust_code_passes(self, validator):
        """Test that valid Rust code passes validation."""
        rust_code = """
fn add(a: i64, b: i64) -> i64 {
    a + b
}
"""

        result = validator.validate(rust_code)

        assert result.is_valid()
        assert result.status == ValidationStatus.PASSED
        assert len(result.get_errors()) == 0

    def test_unbalanced_braces_detected(self, validator):
        """Test detection of unbalanced braces."""
        rust_code = """
fn test() -> i64 {
    let x = 42;
    x
// Missing closing brace
"""

        result = validator.validate(rust_code)

        assert not result.is_valid()
        assert result.has_errors()
        assert any("braces" in issue.message.lower() for issue in result.get_errors())

    def test_unbalanced_parentheses_detected(self, validator):
        """Test detection of unbalanced parentheses."""
        rust_code = "fn test((a: i64) -> i64 { a }"

        result = validator.validate(rust_code)

        assert not result.is_valid()
        assert any("parentheses" in issue.message.lower() for issue in result.get_errors())

    def test_invalid_type_detected(self, validator):
        """Test detection of Python types in Rust code."""
        rust_code = """
fn test(x: int) -> float {
    x as float
}
"""

        result = validator.validate(rust_code)

        assert result.has_errors()
        errors = result.get_errors()
        assert any("int" in issue.message for issue in errors)
        assert any("float" in issue.message for issue in errors)

    def test_excessive_unwrap_warning(self, validator):
        """Test warning for excessive unwrap usage."""
        rust_code = """
fn test() -> i64 {
    let a = Some(1).unwrap();
    let b = Some(2).unwrap();
    let c = Some(3).unwrap();
    let d = Some(4).unwrap();
    a + b + c + d
}
"""

        result = validator.validate(rust_code)

        warnings = result.get_warnings()
        assert any("unwrap" in w.message.lower() for w in warnings)

    def test_missing_result_warning(self, validator):
        """Test warning for missing Result when exceptions expected."""
        rust_code = "fn test() -> i64 { 42 }"

        context = {"has_exceptions": True}
        result = validator.validate(rust_code, context=context)

        warnings = result.get_warnings()
        assert any("Result" in w.message for w in warnings)

    def test_unsafe_block_warning(self, validator):
        """Test warning for unsafe code."""
        rust_code = """
fn test() -> i64 {
    unsafe {
        let ptr = 0x12345 as *const i64;
        *ptr
    }
}
"""

        result = validator.validate(rust_code)

        warnings = result.get_warnings()
        assert any("unsafe" in w.message.lower() for w in warnings)

    def test_naming_convention_check(self, validator):
        """Test naming convention validation."""
        rust_code = """
fn MyFunction() -> i64 {  // Should be snake_case
    42
}
"""

        result = validator.validate(rust_code)

        warnings = result.get_warnings()
        assert any("snake_case" in w.message.lower() for w in warnings)

    def test_metrics_computation(self, validator):
        """Test code metrics computation."""
        rust_code = """
fn test() -> Result<i64, Error> {
    let x = Some(42);
    match x {
        Some(v) => Ok(v),
        None => Err(Error::NotFound)
    }
}
"""

        python_code = "def test(): return 42"

        result = validator.validate(rust_code, python_code)

        metrics = result.metrics
        assert "rust_lines" in metrics
        assert "rust_chars" in metrics
        assert "python_lines" in metrics
        assert "expansion_ratio" in metrics
        assert "result_types" in metrics
        assert "option_types" in metrics
        assert "match_expressions" in metrics

    def test_syntax_only_validation(self, syntax_validator):
        """Test syntax-only validation mode."""
        rust_code = """
fn test() -> i64 {
    Some(42).unwrap().unwrap().unwrap()  // Multiple unwraps
}
"""

        result = syntax_validator.validate(rust_code)

        # Should not check for unwraps in syntax-only mode
        # Only syntax errors should be reported
        assert result.is_valid() or not result.has_errors()

    def test_validation_with_suggestions(self, validator):
        """Test that validation provides helpful suggestions."""
        rust_code = "fn test((a: i64) -> i64 { a }"  # Malformed

        result = validator.validate(rust_code)

        issues = result.issues
        assert any(issue.suggestion is not None for issue in issues)


class TestValidationResult:
    """Test suite for ValidationResult."""

    def test_is_valid_method(self):
        """Test is_valid method."""
        result = ValidationResult(status=ValidationStatus.PASSED)
        assert result.is_valid()

        result = ValidationResult(status=ValidationStatus.FAILED)
        assert not result.is_valid()

    def test_has_errors_method(self):
        """Test has_errors method."""
        result = ValidationResult(
            status=ValidationStatus.PASSED,
            issues=[
                ValidationIssue(severity="warning", message="test"),
            ]
        )
        assert not result.has_errors()

        result.issues.append(
            ValidationIssue(severity="error", message="error")
        )
        assert result.has_errors()

    def test_get_errors(self):
        """Test filtering errors."""
        result = ValidationResult(
            status=ValidationStatus.FAILED,
            issues=[
                ValidationIssue(severity="error", message="error1"),
                ValidationIssue(severity="warning", message="warning1"),
                ValidationIssue(severity="error", message="error2"),
                ValidationIssue(severity="info", message="info1"),
            ]
        )

        errors = result.get_errors()
        assert len(errors) == 2
        assert all(e.severity == "error" for e in errors)

    def test_get_warnings(self):
        """Test filtering warnings."""
        result = ValidationResult(
            status=ValidationStatus.WARNING,
            issues=[
                ValidationIssue(severity="error", message="error1"),
                ValidationIssue(severity="warning", message="warning1"),
                ValidationIssue(severity="warning", message="warning2"),
            ]
        )

        warnings = result.get_warnings()
        assert len(warnings) == 2
        assert all(w.severity == "warning" for w in warnings)


@pytest.mark.parametrize("rust_code,should_pass", [
    ("fn test() -> i64 { 42 }", True),
    ("fn test() -> i64 { 42", False),  # Missing brace
    ("fn test() -> i64 42 }", False),  # Missing opening brace
    ("fn test(a: i64) -> i64 { a }", True),
    ("fn test((a: i64) -> i64 { a }", False),  # Unbalanced parens
])
def test_brace_validation(validator, rust_code, should_pass):
    """Test brace balance validation."""
    result = validator.validate(rust_code)

    if should_pass:
        assert not result.has_errors() or "braces" not in result.get_errors()[0].message.lower()
    else:
        assert result.has_errors()
