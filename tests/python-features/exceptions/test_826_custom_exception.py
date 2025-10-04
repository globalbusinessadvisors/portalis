"""
Feature: 8.2.6 Custom Exception
Category: Exception Handling
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
class MyError(Exception):
    pass

raise MyError("Custom error")
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_826_custom_exception():
    """Test translation of 8.2.6 Custom Exception."""
    pytest.skip("Feature not yet implemented")
