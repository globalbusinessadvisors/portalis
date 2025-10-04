"""
Feature: 2.1.2 Subtraction (-)
Category: Operators
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
result = 5 - 3  # 2
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_212_subtraction():
    """Test translation of 2.1.2 Subtraction (-)."""
    pytest.skip("Feature not yet implemented")
