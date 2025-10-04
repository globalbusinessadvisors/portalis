"""
Feature: 3.2.1 While Loop
Category: Control Flow
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
while x > 0:
    x -= 1
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_321_while_loop():
    """Test translation of 3.2.1 While Loop."""
    pytest.skip("Feature not yet implemented")
