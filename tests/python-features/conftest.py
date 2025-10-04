"""
Pytest configuration for Python language feature tests.

This module configures pytest with:
- Custom markers for complexity and status
- Fixtures for translation testing
- Coverage tracking
- Report generation
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "complexity_low: Features with direct Rust equivalents"
    )
    config.addinivalue_line(
        "markers", "complexity_medium: Features requiring adaptation"
    )
    config.addinivalue_line(
        "markers", "complexity_high: Features requiring significant effort"
    )
    config.addinivalue_line(
        "markers", "complexity_very_high: Extremely difficult/impossible features"
    )
    config.addinivalue_line(
        "markers", "status_not_implemented: Feature translation not yet implemented"
    )
    config.addinivalue_line(
        "markers", "status_partial: Feature partially implemented"
    )
    config.addinivalue_line(
        "markers", "status_implemented: Feature fully implemented"
    )
    config.addinivalue_line(
        "markers", "status_unsupported: Feature cannot be translated"
    )


@pytest.fixture
def translator():
    """Fixture for Python to Rust translator (placeholder)."""
    # TODO: Replace with actual translator when implemented
    class MockTranslator:
        def translate(self, python_code: str) -> str:
            """Mock translation - returns empty Rust code."""
            return "// Translation not yet implemented"

    return MockTranslator()


@pytest.fixture
def rust_compiler():
    """Fixture for Rust compiler integration (placeholder)."""
    # TODO: Replace with actual compiler when implemented
    class MockCompiler:
        def compile(self, rust_code: str) -> bool:
            """Mock compilation - always fails for now."""
            return False

        def compile_and_run(self, rust_code: str) -> str:
            """Mock execution - returns empty string."""
            return ""

    return MockCompiler()


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file metadata."""
    for item in items:
        # Read test file to extract metadata
        test_file = Path(item.fspath)
        if test_file.exists():
            content = test_file.read_text()

            # Extract complexity from docstring
            if "Complexity: Low" in content:
                item.add_marker(pytest.mark.complexity_low)
            elif "Complexity: Medium" in content:
                item.add_marker(pytest.mark.complexity_medium)
            elif "Complexity: High" in content:
                item.add_marker(pytest.mark.complexity_high)
            elif "Complexity: Very High" in content:
                item.add_marker(pytest.mark.complexity_very_high)

            # Extract status from docstring
            if "Status: not_implemented" in content:
                item.add_marker(pytest.mark.status_not_implemented)
            elif "Status: partial" in content:
                item.add_marker(pytest.mark.status_partial)
            elif "Status: implemented" in content:
                item.add_marker(pytest.mark.status_implemented)
            elif "Status: unsupported" in content:
                item.add_marker(pytest.mark.status_unsupported)


def pytest_sessionstart(session):
    """Called before test session starts."""
    session.coverage_tracker = {
        'start_time': datetime.now().isoformat(),
        'total_features': 0,
        'implemented': 0,
        'partial': 0,
        'not_implemented': 0,
        'unsupported': 0,
        'by_complexity': {
            'low': {'total': 0, 'implemented': 0},
            'medium': {'total': 0, 'implemented': 0},
            'high': {'total': 0, 'implemented': 0},
            'very_high': {'total': 0, 'implemented': 0},
        },
        'by_category': {},
    }


def pytest_sessionfinish(session, exitstatus):
    """Called after test session finishes - generate coverage report."""
    if hasattr(session, 'coverage_tracker'):
        report_path = Path(__file__).parent / 'coverage_report.json'

        # Add finish time
        session.coverage_tracker['end_time'] = datetime.now().isoformat()

        # Calculate percentages
        total = session.coverage_tracker['total_features']
        if total > 0:
            session.coverage_tracker['coverage_percentage'] = round(
                (session.coverage_tracker['implemented'] / total) * 100, 2
            )

        # Write report
        with open(report_path, 'w') as f:
            json.dump(session.coverage_tracker, f, indent=2)

        print(f"\n📊 Coverage report written to: {report_path}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Track test results for coverage reporting."""
    outcome = yield
    rep = outcome.get_result()

    if rep.when == 'call':
        # Update coverage tracker
        session = item.session
        if hasattr(session, 'coverage_tracker'):
            tracker = session.coverage_tracker

            # Count total features
            tracker['total_features'] += 1

            # Get category from test path
            test_path = Path(item.fspath)
            category = test_path.parent.name
            if category not in tracker['by_category']:
                tracker['by_category'][category] = {
                    'total': 0,
                    'implemented': 0,
                    'partial': 0,
                }
            tracker['by_category'][category]['total'] += 1

            # Track complexity
            for marker in item.iter_markers():
                if marker.name.startswith('complexity_'):
                    complexity = marker.name.replace('complexity_', '')
                    if complexity in tracker['by_complexity']:
                        tracker['by_complexity'][complexity]['total'] += 1

                # Track implementation status
                if marker.name == 'status_implemented':
                    tracker['implemented'] += 1
                    tracker['by_category'][category]['implemented'] += 1
                    # Update complexity tracking
                    for comp_marker in item.iter_markers():
                        if comp_marker.name.startswith('complexity_'):
                            comp = comp_marker.name.replace('complexity_', '')
                            if comp in tracker['by_complexity']:
                                tracker['by_complexity'][comp]['implemented'] += 1

                elif marker.name == 'status_partial':
                    tracker['partial'] += 1
                    tracker['by_category'][category]['partial'] += 1

                elif marker.name == 'status_not_implemented':
                    tracker['not_implemented'] += 1

                elif marker.name == 'status_unsupported':
                    tracker['unsupported'] += 1
