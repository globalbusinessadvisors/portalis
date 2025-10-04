#!/usr/bin/env python3
"""
Generate test stub files for all 527 Python language features.

This script reads PYTHON_LANGUAGE_FEATURES.md and creates a test file
for each feature in the appropriate category directory.
"""

import os
import re
from pathlib import Path

# Category mapping: markdown section name -> directory name
CATEGORY_MAP = {
    "Basic Syntax & Literals": "basic-syntax",
    "Operators": "operators",
    "Control Flow": "control-flow",
    "Data Structures": "data-structures",
    "Functions": "functions",
    "Classes & OOP": "classes",
    "Modules & Imports": "modules",
    "Exception Handling": "exceptions",
    "Context Managers": "context-managers",
    "Iterators & Generators": "iterators",
    "Async/Await": "async-await",
    "Type Hints": "type-hints",
    "Metaclasses": "metaclasses",
    "Descriptors": "descriptors",
    "Built-in Functions": "builtins",
    "Magic Methods": "magic-methods",
}

TEST_STUB_TEMPLATE = '''"""
Feature: {feature_name}
Category: {category}
Complexity: {complexity}
Status: not_implemented
"""

import pytest

PYTHON_SOURCE = """
{python_code}
"""

EXPECTED_RUST = """
{rust_code}
"""

@pytest.mark.complexity("{complexity_lower}")
@pytest.mark.status("not_implemented")
def test_{test_name}():
    """Test translation of {feature_name}."""
    pytest.skip("Feature not yet implemented")
'''


def sanitize_filename(name: str) -> str:
    """Convert feature name to valid filename."""
    # Remove special characters, convert to lowercase
    name = re.sub(r'[^a-zA-Z0-9\s_-]', '', name)
    name = name.lower().replace(' ', '_').replace('-', '_')
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def parse_features(md_file: Path):
    """Parse PYTHON_LANGUAGE_FEATURES.md and extract features."""
    content = md_file.read_text()

    features = []
    current_category = None
    current_feature = None
    current_complexity = None
    python_code = []
    rust_code = []
    in_python_block = False
    in_rust_block = False

    for line in content.split('\n'):
        # Detect category headers
        if line.startswith('## ') and any(cat in line for cat in CATEGORY_MAP.keys()):
            for cat_name in CATEGORY_MAP.keys():
                if cat_name in line:
                    current_category = cat_name
                    break

        # Detect feature headers
        elif line.startswith('#### '):
            # Save previous feature if exists
            if current_feature and current_category:
                features.append({
                    'category': current_category,
                    'name': current_feature,
                    'complexity': current_complexity or 'Low',
                    'python': '\n'.join(python_code).strip(),
                    'rust': '\n'.join(rust_code).strip(),
                })

            # Start new feature
            current_feature = line.replace('####', '').strip()
            python_code = []
            rust_code = []
            in_python_block = False
            in_rust_block = False

        # Detect complexity
        elif line.startswith('**Complexity**:'):
            current_complexity = line.split(':')[1].strip()

        # Detect Python code blocks
        elif '```python' in line:
            in_python_block = True
            in_rust_block = False
        elif '```rust' in line:
            in_rust_block = True
            in_python_block = False
        elif line.startswith('```') and (in_python_block or in_rust_block):
            in_python_block = False
            in_rust_block = False
        elif in_python_block:
            python_code.append(line)
        elif in_rust_block:
            rust_code.append(line)

    # Save last feature
    if current_feature and current_category:
        features.append({
            'category': current_category,
            'name': current_feature,
            'complexity': current_complexity or 'Low',
            'python': '\n'.join(python_code).strip(),
            'rust': '\n'.join(rust_code).strip(),
        })

    return features


def generate_test_files(features, base_dir: Path):
    """Generate test stub files for all features."""
    stats = {cat: 0 for cat in CATEGORY_MAP.values()}

    for feature in features:
        category_dir = CATEGORY_MAP.get(feature['category'])
        if not category_dir:
            print(f"⚠️  Unknown category: {feature['category']}")
            continue

        # Create test filename
        test_name = sanitize_filename(feature['name'])
        test_file = base_dir / category_dir / f"test_{test_name}.py"

        # Generate test content
        content = TEST_STUB_TEMPLATE.format(
            feature_name=feature['name'],
            category=feature['category'],
            complexity=feature['complexity'],
            complexity_lower=feature['complexity'].lower(),
            test_name=test_name,
            python_code=feature['python'] or '# TODO: Add Python example',
            rust_code=feature['rust'] or '// TODO: Add Rust equivalent',
        )

        # Write file
        test_file.write_text(content)
        stats[category_dir] += 1
        print(f"✓ Created {test_file.relative_to(base_dir)}")

    return stats


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    features_file = repo_root / 'PYTHON_LANGUAGE_FEATURES.md'
    test_base_dir = repo_root / 'tests' / 'python-features'

    if not features_file.exists():
        print(f"❌ Feature catalog not found: {features_file}")
        return 1

    print(f"📖 Parsing feature catalog: {features_file}")
    features = parse_features(features_file)
    print(f"✓ Found {len(features)} features\n")

    print(f"📝 Generating test stubs in {test_base_dir}...")
    stats = generate_test_files(features, test_base_dir)

    print(f"\n✅ Test stub generation complete!")
    print(f"\nFiles created per category:")
    total = 0
    for category, count in sorted(stats.items()):
        print(f"  {category:20s}: {count:3d} tests")
        total += count
    print(f"  {'TOTAL':20s}: {total:3d} tests")

    return 0


if __name__ == '__main__':
    exit(main())
