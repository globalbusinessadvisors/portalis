"""
Feature: 1.5.2 Underscore (throwaway variable)
Category: Basic Syntax & Literals
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
_, x, _ = (1, 2, 3)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_152_underscore_throwaway_variable():
    """Test translation of 1.5.2 Underscore (throwaway variable)."""
    pytest.skip("Feature not yet implemented")
