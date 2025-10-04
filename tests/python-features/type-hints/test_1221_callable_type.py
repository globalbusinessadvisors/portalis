"""
Feature: 12.2.1 Callable Type
Category: Type Hints
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
from typing import Callable
func: Callable[[int, int], int] = lambda a, b: a + b
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_1221_callable_type():
    """Test translation of 12.2.1 Callable Type."""
    pytest.skip("Feature not yet implemented")
