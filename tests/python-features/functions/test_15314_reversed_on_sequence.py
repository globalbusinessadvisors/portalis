"""
Feature: 15.3.14 reversed() on sequence
Category: Functions
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
for item in reversed([1, 2, 3]):
    pass
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_15314_reversed_on_sequence():
    """Test translation of 15.3.14 reversed() on sequence."""
    pytest.skip("Feature not yet implemented")
