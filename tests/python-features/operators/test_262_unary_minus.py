"""
Feature: 2.6.2 Unary Minus (-)
Category: Operators
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
result = -5  # -5
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_262_unary_minus():
    """Test translation of 2.6.2 Unary Minus (-)."""
    pytest.skip("Feature not yet implemented")
