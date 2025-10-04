"""
Feature: 8.2.1 Raise Exception
Category: Exception Handling
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
raise ValueError("Invalid value")
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_821_raise_exception():
    """Test translation of 8.2.1 Raise Exception."""
    pytest.skip("Feature not yet implemented")
