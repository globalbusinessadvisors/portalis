"""
Translation Validation

Validates generated Rust code for syntactic correctness,
semantic equivalence, and quality metrics.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
from loguru import logger


class ValidationLevel(Enum):
    """Validation strictness levels."""
    SYNTAX_ONLY = "syntax"
    SEMANTIC = "semantic"
    FULL = "full"


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    severity: str  # "error", "warning", "info"
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating translated code."""

    status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASSED

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(issue.severity == "error" for issue in self.issues)

    def get_errors(self) -> List[ValidationIssue]:
        """Get all error issues."""
        return [issue for issue in self.issues if issue.severity == "error"]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning issues."""
        return [issue for issue in self.issues if issue.severity == "warning"]


class TranslationValidator:
    """
    Validator for translated Rust code.

    Performs:
    - Syntax validation
    - Semantic correctness checks
    - Code quality metrics
    - Best practices verification
    """

    def __init__(self, level: ValidationLevel = ValidationLevel.FULL):
        self.level = level
        self.checks = {
            "syntax": [
                self._check_balanced_braces,
                self._check_function_syntax,
                self._check_type_syntax,
            ],
            "semantic": [
                self._check_error_handling,
                self._check_ownership_patterns,
                self._check_unsafe_blocks,
            ],
            "quality": [
                self._check_code_formatting,
                self._check_naming_conventions,
                self._check_documentation,
            ]
        }

    def validate(
        self,
        rust_code: str,
        python_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate translated Rust code.

        Args:
            rust_code: Generated Rust code
            python_code: Original Python code (for semantic comparison)
            context: Additional validation context

        Returns:
            ValidationResult with status and issues
        """
        result = ValidationResult(
            status=ValidationStatus.PASSED,
            issues=[],
            metrics={}
        )

        context = context or {}

        # Run syntax checks
        for check in self.checks["syntax"]:
            check_name = check.__name__
            try:
                issues = check(rust_code, context)
                if issues:
                    result.issues.extend(issues)
                    result.failed_checks.append(check_name)
                else:
                    result.passed_checks.append(check_name)
            except Exception as e:
                logger.error(f"Check {check_name} failed: {e}")
                result.issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Validation check failed: {check_name}",
                    )
                )

        # Run semantic checks if level permits
        if self.level in [ValidationLevel.SEMANTIC, ValidationLevel.FULL]:
            for check in self.checks["semantic"]:
                check_name = check.__name__
                try:
                    issues = check(rust_code, context)
                    if issues:
                        result.issues.extend(issues)
                        result.failed_checks.append(check_name)
                    else:
                        result.passed_checks.append(check_name)
                except Exception as e:
                    logger.error(f"Check {check_name} failed: {e}")

        # Run quality checks if full validation
        if self.level == ValidationLevel.FULL:
            for check in self.checks["quality"]:
                check_name = check.__name__
                try:
                    issues = check(rust_code, context)
                    if issues:
                        result.issues.extend(issues)
                        result.failed_checks.append(check_name)
                    else:
                        result.passed_checks.append(check_name)
                except Exception as e:
                    logger.error(f"Check {check_name} failed: {e}")

        # Compute metrics
        result.metrics = self._compute_metrics(rust_code, python_code)

        # Determine overall status
        if result.has_errors():
            result.status = ValidationStatus.FAILED
        elif result.get_warnings():
            result.status = ValidationStatus.WARNING
        else:
            result.status = ValidationStatus.PASSED

        return result

    def _check_balanced_braces(
        self,
        rust_code: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check if braces are balanced."""
        issues = []

        # Count braces
        open_braces = rust_code.count('{')
        close_braces = rust_code.count('}')

        if open_braces != close_braces:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Unbalanced braces: {open_braces} open, {close_braces} close",
                    suggestion="Check for missing or extra braces"
                )
            )

        # Count parentheses
        open_parens = rust_code.count('(')
        close_parens = rust_code.count(')')

        if open_parens != close_parens:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Unbalanced parentheses: {open_parens} open, {close_parens} close",
                    suggestion="Check for missing or extra parentheses"
                )
            )

        return issues

    def _check_function_syntax(
        self,
        rust_code: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check basic function syntax."""
        issues = []

        # Check for function definitions
        fn_pattern = r'(?:pub\s+)?(?:async\s+)?fn\s+\w+\s*\([^)]*\)'

        functions = re.findall(fn_pattern, rust_code)

        if not functions and 'fn ' in rust_code:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message="Potentially malformed function definition",
                    suggestion="Check function signature syntax"
                )
            )

        return issues

    def _check_type_syntax(
        self,
        rust_code: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check type syntax."""
        issues = []

        # Check for common type errors
        invalid_types = [
            "int", "float", "str", "bool",  # Python types in Rust
        ]

        for invalid in invalid_types:
            pattern = rf'\b{invalid}\b'
            if re.search(pattern, rust_code):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"Invalid Rust type: '{invalid}'",
                        location=invalid,
                        suggestion=f"Use Rust equivalent (e.g., i64, f64, String, bool)"
                    )
                )

        return issues

    def _check_error_handling(
        self,
        rust_code: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check error handling patterns."""
        issues = []

        # Check for unwrap() usage (should be minimal)
        unwrap_count = rust_code.count('.unwrap()')

        if unwrap_count > 2:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Excessive use of .unwrap() ({unwrap_count} times)",
                    suggestion="Consider proper error handling with Result"
                )
            )

        # Check for Result types
        has_result = 'Result<' in rust_code

        if context.get("has_exceptions") and not has_result:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message="Python code raises exceptions but Rust code doesn't use Result",
                    suggestion="Add Result return type for error handling"
                )
            )

        return issues

    def _check_ownership_patterns(
        self,
        rust_code: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check ownership and borrowing patterns."""
        issues = []

        # Check for excessive cloning
        clone_count = rust_code.count('.clone()')

        if clone_count > 5:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Excessive use of .clone() ({clone_count} times)",
                    suggestion="Consider using references to avoid unnecessary clones"
                )
            )

        return issues

    def _check_unsafe_blocks(
        self,
        rust_code: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check for unsafe code blocks."""
        issues = []

        # Count unsafe blocks
        unsafe_count = rust_code.count('unsafe ')

        if unsafe_count > 0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Found {unsafe_count} unsafe block(s)",
                    suggestion="Ensure unsafe code is necessary and well-documented"
                )
            )

        return issues

    def _check_code_formatting(
        self,
        rust_code: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check code formatting."""
        issues = []

        # Check line length
        lines = rust_code.split('\n')
        long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 100]

        if len(long_lines) > len(lines) * 0.2:  # More than 20% long lines
            issues.append(
                ValidationIssue(
                    severity="info",
                    message=f"{len(long_lines)} lines exceed 100 characters",
                    suggestion="Consider running rustfmt"
                )
            )

        return issues

    def _check_naming_conventions(
        self,
        rust_code: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check naming conventions."""
        issues = []

        # Check for snake_case functions
        fn_pattern = r'fn\s+([A-Z]\w*)\s*\('
        camel_case_fns = re.findall(fn_pattern, rust_code)

        if camel_case_fns:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Function names should use snake_case: {camel_case_fns}",
                    suggestion="Rename functions to use snake_case"
                )
            )

        return issues

    def _check_documentation(
        self,
        rust_code: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check documentation."""
        issues = []

        # Check for doc comments
        has_doc_comments = '///' in rust_code or '//!' in rust_code

        if not has_doc_comments and len(rust_code) > 100:
            issues.append(
                ValidationIssue(
                    severity="info",
                    message="No documentation comments found",
                    suggestion="Add /// or //! comments to document code"
                )
            )

        return issues

    def _compute_metrics(
        self,
        rust_code: str,
        python_code: Optional[str]
    ) -> Dict[str, Any]:
        """Compute code quality metrics."""
        metrics = {
            "rust_lines": len(rust_code.split('\n')),
            "rust_chars": len(rust_code),
        }

        if python_code:
            metrics["python_lines"] = len(python_code.split('\n'))
            metrics["python_chars"] = len(python_code)
            metrics["expansion_ratio"] = len(rust_code) / max(len(python_code), 1)

        # Count Rust-specific constructs
        metrics["unsafe_blocks"] = rust_code.count('unsafe ')
        metrics["result_types"] = rust_code.count('Result<')
        metrics["option_types"] = rust_code.count('Option<')
        metrics["match_expressions"] = rust_code.count('match ')

        return metrics
