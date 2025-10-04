"""
Feature: 10.1.3 iter() Function
Category: Iterators & Generators
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
it = iter([1, 2, 3])
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_1013_iter_function():
    """Test translation of 10.1.3 iter() Function."""
    pytest.skip("Feature not yet implemented")
