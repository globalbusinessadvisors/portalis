"""
Feature: 5.6.3 Access __doc__
Category: Functions
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
print(func.__doc__)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_563_access_doc():
    """Test translation of 5.6.3 Access __doc__."""
    pytest.skip("Feature not yet implemented")
