"""
Feature: 1.2.8 Formatted Strings (f-strings)
Category: Basic Syntax & Literals
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
name = "Alice"
age = 30
s = f"My name is {name} and I'm {age} years old"
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_128_formatted_strings_f_strings():
    """Test translation of 1.2.8 Formatted Strings (f-strings)."""
    pytest.skip("Feature not yet implemented")
