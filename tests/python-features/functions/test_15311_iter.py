"""
Feature: 15.3.11 iter()
Category: Functions
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
it = iter(iterable)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_15311_iter():
    """Test translation of 15.3.11 iter()."""
    pytest.skip("Feature not yet implemented")
