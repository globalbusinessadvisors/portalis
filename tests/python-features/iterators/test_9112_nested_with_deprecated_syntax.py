"""
Feature: 9.1.12 Nested With (deprecated syntax)
Category: Iterators & Generators
Complexity: Low
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
with A() as a:
    with B() as b:
        use(a, b)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("low")
@pytest.mark.status("not_implemented")
def test_9112_nested_with_deprecated_syntax():
    """Test translation of 9.1.12 Nested With (deprecated syntax)."""
    pytest.skip("Feature not yet implemented")
