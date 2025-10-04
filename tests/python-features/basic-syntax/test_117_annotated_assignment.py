"""
Feature: 1.1.7 Annotated Assignment
Category: Basic Syntax & Literals
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
count: int = 0
name: str = "Alice"
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_117_annotated_assignment():
    """Test translation of 1.1.7 Annotated Assignment."""
    pytest.skip("Feature not yet implemented")
