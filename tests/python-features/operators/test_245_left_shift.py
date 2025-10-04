"""
Feature: 2.4.5 Left Shift (<<)
Category: Operators
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
result = 5 << 2  # 20
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_245_left_shift():
    """Test translation of 2.4.5 Left Shift (<<)."""
    pytest.skip("Feature not yet implemented")
