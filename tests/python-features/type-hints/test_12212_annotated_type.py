"""
Feature: 12.2.12 Annotated Type
Category: Type Hints
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
from typing import Annotated
Age = Annotated[int, "Must be >= 0"]
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_12212_annotated_type():
    """Test translation of 12.2.12 Annotated Type."""
    pytest.skip("Feature not yet implemented")
