"""
Feature: 3.2.2 While-Else Loop
Category: Control Flow
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
while x > 0:
    x -= 1
else:
    print("Loop completed normally")
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_322_while_else_loop():
    """Test translation of 3.2.2 While-Else Loop."""
    pytest.skip("Feature not yet implemented")
