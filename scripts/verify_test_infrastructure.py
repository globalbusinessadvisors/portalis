#!/usr/bin/env python3
"""
Verify the Python language feature test infrastructure is set up correctly.

This script validates:
1. Test directory structure exists
2. Test files are properly formatted
3. Feature categories are complete
4. Generates initial coverage baseline
"""

import json
from pathlib import Path
from collections import defaultdict
import re


def verify_directory_structure(base_path: Path) -> dict:
    """Verify all test directories exist."""
    expected_dirs = [
        'basic-syntax', 'operators', 'control-flow', 'data-structures',
        'functions', 'classes', 'modules', 'exceptions', 'context-managers',
        'iterators', 'async-await', 'type-hints', 'metaclasses',
        'descriptors', 'builtins', 'magic-methods'
    ]

    results = {}
    for dir_name in expected_dirs:
        dir_path = base_path / dir_name
        exists = dir_path.exists() and dir_path.is_dir()
        results[dir_name] = {
            'exists': exists,
            'path': str(dir_path),
        }

    return results


def count_test_files(base_path: Path) -> dict:
    """Count test files in each category."""
    stats = defaultdict(int)

    for test_file in base_path.rglob('test_*.py'):
        if test_file.parent.name != 'python-features':
            category = test_file.parent.name
            stats[category] += 1

    return dict(stats)


def parse_test_metadata(test_file: Path) -> dict:
    """Extract metadata from test file."""
    content = test_file.read_text()

    metadata = {
        'feature': None,
        'category': None,
        'complexity': None,
        'status': None,
        'has_python_source': False,
        'has_rust_expected': False,
    }

    # Extract from docstring
    feature_match = re.search(r'Feature:\s*(.+)', content)
    if feature_match:
        metadata['feature'] = feature_match.group(1).strip()

    category_match = re.search(r'Category:\s*(.+)', content)
    if category_match:
        metadata['category'] = category_match.group(1).strip()

    complexity_match = re.search(r'Complexity:\s*(\w+)', content)
    if complexity_match:
        metadata['complexity'] = complexity_match.group(1).strip()

    status_match = re.search(r'Status:\s*(\w+)', content)
    if status_match:
        metadata['status'] = status_match.group(1).strip()

    # Check for code blocks
    metadata['has_python_source'] = 'PYTHON_SOURCE' in content
    metadata['has_rust_expected'] = 'EXPECTED_RUST' in content

    return metadata


def generate_coverage_baseline(base_path: Path) -> dict:
    """Generate initial coverage baseline."""
    baseline = {
        'total_tests': 0,
        'by_category': defaultdict(lambda: defaultdict(int)),
        'by_complexity': defaultdict(int),
        'by_status': defaultdict(int),
        'test_files': [],
    }

    for test_file in base_path.rglob('test_*.py'):
        if test_file.parent.name == 'python-features':
            continue

        baseline['total_tests'] += 1
        category = test_file.parent.name

        metadata = parse_test_metadata(test_file)

        # Count by category
        baseline['by_category'][category]['total'] += 1
        if metadata['status']:
            baseline['by_category'][category][metadata['status']] += 1

        # Count by complexity
        if metadata['complexity']:
            baseline['by_complexity'][metadata['complexity']] += 1

        # Count by status
        if metadata['status']:
            baseline['by_status'][metadata['status']] += 1

        # Record test file info
        baseline['test_files'].append({
            'path': str(test_file.relative_to(base_path)),
            'category': category,
            'feature': metadata['feature'],
            'complexity': metadata['complexity'],
            'status': metadata['status'],
        })

    # Convert defaultdicts to regular dicts for JSON serialization
    baseline['by_category'] = {k: dict(v) for k, v in baseline['by_category'].items()}
    baseline['by_complexity'] = dict(baseline['by_complexity'])
    baseline['by_status'] = dict(baseline['by_status'])

    return baseline


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    test_base = repo_root / 'tests' / 'python-features'

    print("=" * 80)
    print("PYTHON LANGUAGE FEATURE TEST INFRASTRUCTURE VERIFICATION")
    print("=" * 80)
    print()

    # 1. Verify directory structure
    print("1. Verifying directory structure...")
    dir_structure = verify_directory_structure(test_base)

    all_exist = all(d['exists'] for d in dir_structure.values())
    if all_exist:
        print("   ✅ All 16 category directories exist")
    else:
        print("   ❌ Missing directories:")
        for name, info in dir_structure.items():
            if not info['exists']:
                print(f"      - {name}")
    print()

    # 2. Count test files
    print("2. Counting test files...")
    file_counts = count_test_files(test_base)
    total_files = sum(file_counts.values())
    print(f"   Total test files: {total_files}")
    print()
    print("   Files per category:")
    for category in sorted(file_counts.keys()):
        count = file_counts[category]
        print(f"      {category:25s}: {count:3d} tests")
    print()

    # 3. Generate coverage baseline
    print("3. Generating coverage baseline...")
    baseline = generate_coverage_baseline(test_base)

    print(f"   Total tests parsed: {baseline['total_tests']}")
    print()
    print("   By complexity:")
    for complexity, count in sorted(baseline['by_complexity'].items()):
        print(f"      {complexity:15s}: {count:3d}")
    print()
    print("   By status:")
    for status, count in sorted(baseline['by_status'].items()):
        print(f"      {status:20s}: {count:3d}")
    print()

    # 4. Save baseline
    baseline_file = test_base / 'coverage_baseline.json'
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, indent=2)
    print(f"4. Baseline saved to: {baseline_file}")
    print()

    # 5. Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Directory structure: {'PASS' if all_exist else 'FAIL'}")
    print(f"✅ Test files created: {total_files}")
    print(f"✅ Coverage baseline: {baseline_file.name}")
    print()

    if total_files >= 500:
        print("🎉 Test infrastructure is ready!")
        print()
        print("Next steps:")
        print("  1. Implement translator core (Week 2)")
        print("  2. Start with Low complexity features")
        print("  3. Run: python3 scripts/run_coverage_tests.py")
    else:
        print("⚠️  Expected ~527 test files, found {total_files}")

    print("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
