"""
Feature: 3.2.3 For Loop (iterating)
Category: Control Flow
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
for item in collection:
    print(item)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_323_for_loop_iterating():
    """Test translation of 3.2.3 For Loop (iterating)."""
    pytest.skip("Feature not yet implemented")
