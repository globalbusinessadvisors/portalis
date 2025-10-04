"""
Feature: 12.1.4 Optional Type
Category: Type Hints
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
from typing import Optional
value: Optional[int] = None
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_1214_optional_type():
    """Test translation of 12.1.4 Optional Type."""
    pytest.skip("Feature not yet implemented")
