"""
Feature: 1.2.7 Raw Strings
Category: Basic Syntax & Literals
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
path = r"C:\Users\name"
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_127_raw_strings():
    """Test translation of 1.2.7 Raw Strings."""
    pytest.skip("Feature not yet implemented")
