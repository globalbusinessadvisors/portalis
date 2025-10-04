"""
Feature: 12.1.2 Function Parameter Annotation
Category: Type Hints
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
def greet(name: str) -> None:
    print(name)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_1212_function_parameter_annotation():
    """Test translation of 12.1.2 Function Parameter Annotation."""
    pytest.skip("Feature not yet implemented")
