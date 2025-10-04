"""
Feature: 8.1.2 Except with Type
Category: Exception Handling
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
try:
    operation()
except ValueError:
    handle_value_error()
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_812_except_with_type():
    """Test translation of 8.1.2 Except with Type."""
    pytest.skip("Feature not yet implemented")
