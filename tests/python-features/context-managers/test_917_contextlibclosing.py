"""
Feature: 9.1.7 contextlib.closing
Category: Context Managers
Complexity: Medium
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
from contextlib import closing
with closing(resource) as r:
    use(r)
"""

EXPECTED_RUST = """
// TODO: Add Rust equivalent
"""

@pytest.mark.complexity("medium")
@pytest.mark.status("not_implemented")
def test_917_contextlibclosing():
    """Test translation of 9.1.7 contextlib.closing."""
    pytest.skip("Feature not yet implemented")
