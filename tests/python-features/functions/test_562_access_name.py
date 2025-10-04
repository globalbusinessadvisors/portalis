"""
Feature: 5.6.2 Access __name__
Category: Functions
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
print(func.__name__)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_562_access_name():
    """Test translation of 5.6.2 Access __name__."""
    pytest.skip("Feature not yet implemented")
