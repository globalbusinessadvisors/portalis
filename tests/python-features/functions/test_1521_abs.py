"""
Feature: 15.2.1 abs()
Category: Functions
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
x = abs(-5)  # 5
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_1521_abs():
    """Test translation of 15.2.1 abs()."""
    pytest.skip("Feature not yet implemented")
