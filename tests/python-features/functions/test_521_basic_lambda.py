"""
Feature: 5.2.1 Basic Lambda
Category: Functions
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
square = lambda x: x ** 2
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_521_basic_lambda():
    """Test translation of 5.2.1 Basic Lambda."""
    pytest.skip("Feature not yet implemented")
