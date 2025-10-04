"""
Feature: 12.3.9 @no_type_check Decorator
Category: Type Hints
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
from typing import no_type_check

@no_type_check
def func():
    ...
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_1239_no_type_check_decorator():
    """Test translation of 12.3.9 @no_type_check Decorator."""
    pytest.skip("Feature not yet implemented")
