#!/usr/bin/env python3
"""
Run Python language feature coverage tests and generate reports.

This script:
1. Runs pytest on all feature tests
2. Generates coverage metrics
3. Creates HTML and JSON reports
4. Provides implementation progress tracking
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict


def run_tests(category: str = None, complexity: str = None) -> int:
    """Run pytest with optional filters."""
    cmd = ['pytest', 'tests/python-features/', '-v', '--tb=short']

    if category:
        cmd.extend(['-k', category])

    if complexity:
        cmd.extend(['-m', f'complexity_{complexity}'])

    print(f"🧪 Running tests: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def generate_coverage_report():
    """Generate detailed coverage report from test results."""
    report_path = Path(__file__).parent.parent / 'tests/python-features/coverage_report.json'

    if not report_path.exists():
        print("⚠️  No coverage report found. Run tests first.")
        return

    with open(report_path) as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("📊 PORTALIS PYTHON LANGUAGE FEATURE COVERAGE REPORT")
    print("=" * 80)
    print(f"Generated: {data.get('end_time', 'Unknown')}\n")

    # Overall summary
    total = data['total_features']
    implemented = data['implemented']
    partial = data['partial']
    not_impl = data['not_implemented']
    unsupported = data['unsupported']

    print("OVERALL SUMMARY")
    print("-" * 80)
    print(f"  Total Features:          {total:4d}")
    print(f"  ✅ Implemented:          {implemented:4d} ({_percent(implemented, total):5.1f}%)")
    print(f"  🟡 Partially Implemented: {partial:4d} ({_percent(partial, total):5.1f}%)")
    print(f"  ⏳ Not Implemented:      {not_impl:4d} ({_percent(not_impl, total):5.1f}%)")
    print(f"  ❌ Unsupported:          {unsupported:4d} ({_percent(unsupported, total):5.1f}%)")
    print(f"\n  📈 Coverage:             {data.get('coverage_percentage', 0):5.1f}%")

    # Complexity breakdown
    print("\n\nBY COMPLEXITY LEVEL")
    print("-" * 80)
    print(f"{'Complexity':<15} {'Total':>8} {'Implemented':>12} {'Coverage':>10}")
    print("-" * 80)

    for complexity in ['low', 'medium', 'high', 'very_high']:
        stats = data['by_complexity'][complexity]
        total_comp = stats['total']
        impl_comp = stats['implemented']
        coverage = _percent(impl_comp, total_comp)

        print(f"{complexity.replace('_', ' ').title():<15} {total_comp:>8} {impl_comp:>12} {coverage:>9.1f}%")

    # Category breakdown
    print("\n\nBY CATEGORY")
    print("-" * 80)
    print(f"{'Category':<25} {'Total':>8} {'Implemented':>12} {'Partial':>8} {'Coverage':>10}")
    print("-" * 80)

    for category, stats in sorted(data['by_category'].items()):
        total_cat = stats['total']
        impl_cat = stats['implemented']
        partial_cat = stats['partial']
        coverage = _percent(impl_cat, total_cat)

        print(f"{category:<25} {total_cat:>8} {impl_cat:>12} {partial_cat:>8} {coverage:>9.1f}%")

    # Implementation priorities
    print("\n\nIMPLEMENTATION PRIORITIES")
    print("-" * 80)
    print("Recommended implementation order based on complexity and impact:")
    print()
    print("1. Low Complexity (241 features, 45.7%)")
    print("   - Direct Rust equivalents")
    print("   - Quick wins for coverage improvement")
    print()
    print("2. Medium Complexity (159 features, 30.2%)")
    print("   - Requires adaptation but feasible")
    print("   - Essential for real-world Python code")
    print()
    print("3. High Complexity (91 features, 17.3%)")
    print("   - Decorators, generators, metaclasses")
    print("   - Advanced Python features")
    print()
    print("4. Very High Complexity (36 features, 6.8%)")
    print("   - eval(), exec(), dynamic features")
    print("   - May require runtime interpreter")

    print("\n" + "=" * 80)


def _percent(value: int, total: int) -> float:
    """Calculate percentage, handling zero division."""
    return (value / total * 100) if total > 0 else 0.0


def generate_html_report():
    """Generate HTML coverage report (placeholder)."""
    print("\n📄 HTML report generation not yet implemented")
    print("   Will be added in Week 2")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run Python feature coverage tests')
    parser.add_argument('--category', help='Filter by category (e.g., basic-syntax)')
    parser.add_argument('--complexity', choices=['low', 'medium', 'high', 'very_high'],
                        help='Filter by complexity level')
    parser.add_argument('--report-only', action='store_true',
                        help='Skip tests, only generate report')
    parser.add_argument('--html', action='store_true',
                        help='Generate HTML report')

    args = parser.parse_args()

    if not args.report_only:
        # Run tests
        exit_code = run_tests(category=args.category, complexity=args.complexity)

        if exit_code != 0:
            print(f"\n⚠️  Tests exited with code {exit_code}")

    # Generate reports
    generate_coverage_report()

    if args.html:
        generate_html_report()

    return 0


if __name__ == '__main__':
    sys.exit(main())
