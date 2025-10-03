"""
Python Exception to Rust Error Mapping

Maps Python exception hierarchies to Rust error types using Result and custom error enums.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class ErrorHandlingStrategy(Enum):
    """Strategy for handling Python exceptions in Rust."""
    RESULT = "result"  # Return Result<T, E>
    PANIC = "panic"    # Use panic! for unrecoverable errors
    OPTION = "option"  # Return Option<T> for missing values


@dataclass
class ExceptionMapping:
    """Maps a Python exception to Rust error handling."""

    python_exception: str
    rust_error_variant: str
    strategy: ErrorHandlingStrategy
    error_message_template: str
    documentation: str = ""
    requires_imports: Set[str] = field(default_factory=set)


class ErrorMapper:
    """
    Maps Python exceptions to Rust error types.

    Provides:
    - Standard exception mappings
    - Custom exception handling
    - Error enum generation
    - Result type wrapping
    """

    def __init__(self):
        self._standard_exceptions: Dict[str, ExceptionMapping] = {}
        self._custom_exceptions: Dict[str, ExceptionMapping] = {}

        self._initialize_standard_mappings()

    def _initialize_standard_mappings(self) -> None:
        """Initialize mappings for Python built-in exceptions."""

        self._standard_exceptions = {
            # Value errors
            "ValueError": ExceptionMapping(
                python_exception="ValueError",
                rust_error_variant="ValueError",
                strategy=ErrorHandlingStrategy.RESULT,
                error_message_template='Error::ValueError("{}")',
                documentation="Invalid value provided"
            ),

            # Type errors
            "TypeError": ExceptionMapping(
                python_exception="TypeError",
                rust_error_variant="TypeError",
                strategy=ErrorHandlingStrategy.RESULT,
                error_message_template='Error::TypeError {{ expected: "{}", got: "{}" }}',
                documentation="Type mismatch"
            ),

            # Key errors
            "KeyError": ExceptionMapping(
                python_exception="KeyError",
                rust_error_variant="KeyError",
                strategy=ErrorHandlingStrategy.RESULT,
                error_message_template='Error::KeyError("{}")',
                documentation="Key not found in dictionary"
            ),

            # Index errors
            "IndexError": ExceptionMapping(
                python_exception="IndexError",
                rust_error_variant="IndexError",
                strategy=ErrorHandlingStrategy.RESULT,
                error_message_template='Error::IndexError("{}")',
                documentation="Index out of bounds"
            ),

            # Attribute errors
            "AttributeError": ExceptionMapping(
                python_exception="AttributeError",
                rust_error_variant="AttributeError",
                strategy=ErrorHandlingStrategy.RESULT,
                error_message_template='Error::AttributeError("{}")',
                documentation="Attribute not found"
            ),

            # IO errors
            "IOError": ExceptionMapping(
                python_exception="IOError",
                rust_error_variant="IoError",
                strategy=ErrorHandlingStrategy.RESULT,
                error_message_template="Error::IoError(e)",
                documentation="IO operation failed",
                requires_imports={"use std::io;"}
            ),

            "FileNotFoundError": ExceptionMapping(
                python_exception="FileNotFoundError",
                rust_error_variant="IoError",
                strategy=ErrorHandlingStrategy.RESULT,
                error_message_template='Error::IoError(io::Error::new(io::ErrorKind::NotFound, "{}"))',
                documentation="File not found",
                requires_imports={"use std::io;"}
            ),

            # Assertion errors
            "AssertionError": ExceptionMapping(
                python_exception="AssertionError",
                rust_error_variant="AssertionError",
                strategy=ErrorHandlingStrategy.PANIC,
                error_message_template='panic!("Assertion failed: {}")',
                documentation="Assertion failed"
            ),

            # Not implemented
            "NotImplementedError": ExceptionMapping(
                python_exception="NotImplementedError",
                rust_error_variant="NotImplemented",
                strategy=ErrorHandlingStrategy.PANIC,
                error_message_template='panic!("Not implemented: {}")',
                documentation="Feature not implemented"
            ),

            # Runtime errors
            "RuntimeError": ExceptionMapping(
                python_exception="RuntimeError",
                rust_error_variant="RuntimeError",
                strategy=ErrorHandlingStrategy.RESULT,
                error_message_template='Error::RuntimeError("{}")',
                documentation="Runtime error occurred"
            ),

            # Overflow errors
            "OverflowError": ExceptionMapping(
                python_exception="OverflowError",
                rust_error_variant="OverflowError",
                strategy=ErrorHandlingStrategy.RESULT,
                error_message_template='Error::OverflowError("{}")',
                documentation="Numeric overflow"
            ),

            # Zero division
            "ZeroDivisionError": ExceptionMapping(
                python_exception="ZeroDivisionError",
                rust_error_variant="ZeroDivision",
                strategy=ErrorHandlingStrategy.RESULT,
                error_message_template='Error::ZeroDivision',
                documentation="Division by zero"
            ),

            # Stop iteration
            "StopIteration": ExceptionMapping(
                python_exception="StopIteration",
                rust_error_variant="StopIteration",
                strategy=ErrorHandlingStrategy.OPTION,
                error_message_template="None",
                documentation="Iterator exhausted"
            ),
        }

    def get_mapping(self, exception_name: str) -> Optional[ExceptionMapping]:
        """Get mapping for a Python exception."""
        # Check standard exceptions first
        if exception_name in self._standard_exceptions:
            return self._standard_exceptions[exception_name]

        # Check custom exceptions
        if exception_name in self._custom_exceptions:
            return self._custom_exceptions[exception_name]

        # Unknown exception - use generic error
        logger.warning(f"Unknown exception '{exception_name}', using generic Error variant")
        return ExceptionMapping(
            python_exception=exception_name,
            rust_error_variant="GenericError",
            strategy=ErrorHandlingStrategy.RESULT,
            error_message_template=f'Error::GenericError("{exception_name}: {{}}")',
            documentation=f"Custom exception: {exception_name}"
        )

    def register_custom_exception(
        self,
        python_name: str,
        rust_variant: str,
        strategy: ErrorHandlingStrategy = ErrorHandlingStrategy.RESULT,
        **kwargs
    ) -> None:
        """Register a custom exception mapping."""
        mapping = ExceptionMapping(
            python_exception=python_name,
            rust_error_variant=rust_variant,
            strategy=strategy,
            **kwargs
        )
        self._custom_exceptions[python_name] = mapping
        logger.info(f"Registered custom exception: {python_name} -> {rust_variant}")

    def generate_error_enum(
        self,
        exception_names: List[str],
        enum_name: str = "Error"
    ) -> str:
        """
        Generate Rust error enum definition for a set of exceptions.

        Args:
            exception_names: List of Python exception names
            enum_name: Name for the Rust error enum

        Returns:
            Rust code for error enum
        """
        # Collect unique error variants
        variants = set()
        imports = set()

        for exc_name in exception_names:
            mapping = self.get_mapping(exc_name)
            if mapping:
                variants.add((mapping.rust_error_variant, mapping.documentation))
                imports.update(mapping.requires_imports)

        # Generate enum definition
        imports_code = "\n".join(sorted(imports))

        variants_code = []
        for variant, doc in sorted(variants):
            if doc:
                variants_code.append(f"    /// {doc}")

            # Determine variant data
            if "ValueError" in variant or "TypeError" in variant:
                variants_code.append(f"    {variant}(String),")
            elif "IoError" in variant:
                variants_code.append(f"    {variant}(std::io::Error),")
            else:
                variants_code.append(f"    {variant},")

        enum_code = f"""
{imports_code}

#[derive(Debug, thiserror::Error)]
pub enum {enum_name} {{
{chr(10).join(variants_code)}
}}
"""

        return enum_code.strip()

    def translate_raise_statement(
        self,
        exception_name: str,
        error_message: Optional[str] = None
    ) -> str:
        """
        Translate Python raise statement to Rust.

        Args:
            exception_name: Name of exception being raised
            error_message: Optional error message

        Returns:
            Rust code for error handling
        """
        mapping = self.get_mapping(exception_name)
        if not mapping:
            return f'return Err(Error::GenericError("{exception_name}".to_string()));'

        if mapping.strategy == ErrorHandlingStrategy.PANIC:
            msg = error_message or exception_name
            return f'panic!("{msg}");'

        elif mapping.strategy == ErrorHandlingStrategy.OPTION:
            return "return None;"

        else:  # RESULT
            template = mapping.error_message_template
            if error_message and "{}" in template:
                error_code = template.replace("{}", f'"{error_message}"')
            else:
                error_code = template

            return f"return Err({error_code});"

    def translate_try_except(
        self,
        exception_handlers: List[tuple[str, str]]
    ) -> str:
        """
        Translate Python try/except to Rust match expression.

        Args:
            exception_handlers: List of (exception_name, handler_code) tuples

        Returns:
            Rust match expression code
        """
        match_arms = []

        for exc_name, handler_code in exception_handlers:
            mapping = self.get_mapping(exc_name)
            if mapping:
                variant = mapping.rust_error_variant

                # Generate match arm
                if "ValueError" in variant or "TypeError" in variant:
                    pattern = f"Error::{variant}(msg)"
                    arm = f"        {pattern} => {{\n            {handler_code}\n        }}"
                elif "IoError" in variant:
                    pattern = f"Error::{variant}(e)"
                    arm = f"        {pattern} => {{\n            {handler_code}\n        }}"
                else:
                    pattern = f"Error::{variant}"
                    arm = f"        {pattern} => {{\n            {handler_code}\n        }}"

                match_arms.append(arm)

        # Add catch-all
        match_arms.append("        e => return Err(e),")

        match_code = f"""match result {{
    Ok(val) => val,
{chr(10).join(match_arms)}
}}"""

        return match_code

    def wrap_function_result(
        self,
        return_type: str,
        can_raise: bool,
        exception_types: Optional[List[str]] = None
    ) -> str:
        """
        Determine Rust return type for function that can raise exceptions.

        Args:
            return_type: Original Rust return type
            can_raise: Whether function can raise exceptions
            exception_types: List of exceptions that can be raised

        Returns:
            Wrapped return type (e.g., Result<T, Error>)
        """
        if not can_raise:
            return return_type

        # Check if all exceptions use Option strategy
        if exception_types:
            all_option = all(
                self.get_mapping(exc).strategy == ErrorHandlingStrategy.OPTION
                for exc in exception_types
                if self.get_mapping(exc)
            )
            if all_option:
                return f"Option<{return_type}>"

        # Default to Result
        return f"Result<{return_type}, Error>"
