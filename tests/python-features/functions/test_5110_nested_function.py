"""
Feature: 5.1.10 Nested Function
Category: Functions
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
def outer():
    def inner():
        print("Inner")
    inner()
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_5110_nested_function():
    """Test translation of 5.1.10 Nested Function."""
    pytest.skip("Feature not yet implemented")
