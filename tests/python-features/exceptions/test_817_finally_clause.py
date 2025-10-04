"""
Feature: 8.1.7 Finally Clause
Category: Exception Handling
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
try:
    operation()
finally:
    cleanup()
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_817_finally_clause():
    """Test translation of 8.1.7 Finally Clause."""
    pytest.skip("Feature not yet implemented")
