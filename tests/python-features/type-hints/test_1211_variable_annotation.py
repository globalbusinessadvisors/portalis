"""
Feature: 12.1.1 Variable Annotation
Category: Type Hints
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
count: int = 0
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_1211_variable_annotation():
    """Test translation of 12.1.1 Variable Annotation."""
    pytest.skip("Feature not yet implemented")
