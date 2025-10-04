"""
Feature: 12.2.9 ClassVar
Category: Type Hints
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
from typing import ClassVar
class MyClass:
    count: ClassVar[int] = 0
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_1229_classvar():
    """Test translation of 12.2.9 ClassVar."""
    pytest.skip("Feature not yet implemented")
