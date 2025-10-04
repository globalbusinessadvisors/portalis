# Portalis Assessment Tool

## Overview

The Portalis Assessment Tool helps you evaluate Python codebases for compatibility with the Portalis Python→Rust→WASM translation platform. It provides detailed reports on translatability, identifies blockers, estimates migration effort, and generates actionable recommendations.

## Features

- **Feature Detection**: Automatically scans Python code to identify language features, patterns, and dependencies
- **Compatibility Analysis**: Evaluates detected features against Portalis support matrix
- **Effort Estimation**: Calculates migration time and cost based on codebase size and complexity
- **Migration Planning**: Generates phased migration plans with dependency analysis
- **Professional Reports**: Outputs HTML, JSON, Markdown, and PDF reports

## Quick Start

### Assess a Python Project

```bash
portalis assess --project /path/to/python/project --report compatibility.html
```

This will:
1. Scan all `.py` files in the project
2. Analyze detected features for compatibility
3. Calculate translatability score
4. Generate an HTML report

### Generate a Migration Plan

```bash
portalis plan --project /path/to/python/project --strategy bottom-up --output plan.md
```

This creates a phased migration plan based on dependency analysis.

## Commands

### `portalis assess`

Analyzes a Python codebase for Portalis compatibility.

**Options:**
- `-p, --project <PATH>`: Path to Python project directory (required)
- `-r, --report <FILE>`: Output report file (default: `portalis-assessment.html`)
- `-f, --format <FORMAT>`: Report format: `html`, `json`, `markdown`, or `pdf` (default: `html`)
- `-v, --verbose`: Show detailed output during analysis

**Example:**
```bash
portalis assess \
    --project ./my-python-app \
    --report my-app-assessment.html \
    --format html \
    --verbose
```

**Output:**
- Translatability percentage
- Feature breakdown (supported/partial/unsupported)
- Critical blockers list
- Effort estimate (hours/weeks)
- Executive summary with recommendations

### `portalis plan`

Generates a migration plan with phased approach.

**Options:**
- `-p, --project <PATH>`: Path to Python project directory (required)
- `-s, --strategy <STRATEGY>`: Migration strategy (default: `bottom-up`)
  - `full`: Translate everything at once
  - `incremental`: Batch migration (5 modules at a time)
  - `bottom-up`: Dependencies first (safest)
  - `top-down`: Top-level modules first
  - `critical-path`: High-impact modules first
- `-o, --output <FILE>`: Output plan file (Markdown)
- `-v, --verbose`: Show all modules in each phase

**Example:**
```bash
portalis plan \
    --project ./my-python-app \
    --strategy bottom-up \
    --output migration-plan.md \
    --verbose
```

**Output:**
- Migration strategy explanation
- Phased module list
- Estimated duration per phase
- Dependencies and risks
- Recommendations for each strategy

## Understanding the Assessment Report

### Executive Summary

High-level overview for stakeholders:
- **Recommendation**: Whether migration is recommended
- **Translatability**: Overall compatibility percentage
- **Estimated Time**: Project duration (weeks/months)
- **Estimated Cost**: Budget range (based on $150/hour)
- **Key Risks**: Top blockers that must be addressed
- **Key Benefits**: Expected improvements from migration

### Compatibility Score

Detailed breakdown:
- **Overall Score**: Weighted average (full=1.0, partial=0.5, none=0.0)
- **Fully Supported**: Features that translate perfectly
- **Partially Supported**: Features with limitations
- **Unsupported (Blockers)**: Features that prevent translation

### Translation Blockers

Critical issues that must be resolved:

**Impact Levels:**
- **Critical**: Prevents translation entirely (e.g., metaclasses, `eval()`)
- **High**: Prevents specific modules (e.g., `__getattr__`)
- **Medium**: Requires refactoring (e.g., abstract methods)
- **Low**: Minor workarounds needed

Each blocker includes:
- Feature name and occurrence count
- Description of the issue
- Suggested workaround (if available)
- File locations

### Effort Estimate

Time and cost breakdown:
- **Analysis**: Initial investigation (2-3% of total)
- **Refactoring**: Removing blockers (varies by compatibility)
- **Translation**: Core migration work (40-50% of total)
- **Testing**: Validation and bug fixes (30-40% of total)
- **Integration**: Deployment setup (5-10% of total)
- **Documentation**: API docs and guides (5-10% of total)

### Migration Timeline

Phased delivery plan:
1. **Analysis & Planning**: Assessment and strategy
2. **Refactoring**: Remove blockers (if needed)
3. **Translation**: Python → Rust → WASM
4. **Testing & Validation**: Ensure correctness
5. **Integration**: Deploy to production
6. **Documentation**: User and developer docs

## Feature Support Matrix

### Fully Supported (✅)

- Functions with type hints
- Classes with `__init__`
- Basic methods (`__str__`, `__repr__`, `__eq__`, etc.)
- Standard decorators (`@property`, `@staticmethod`, `@classmethod`)
- Arithmetic operators (`__add__`, `__sub__`, `__mul__`, etc.)
- Collection access (`__len__`, `__getitem__`, `__setitem__`)
- Most standard library imports

### Partially Supported (⚠️)

- Async/await (basic patterns only)
- Dataclasses (simple fields only)
- Context managers (`__enter__`, `__exit__`)
- Callable classes (`__call__`)
- Generic types (limited)
- Some decorators (`@lru_cache`)

### Not Supported (❌)

- **Metaclasses**: Requires runtime introspection
- **Dynamic execution**: `eval()`, `exec()`, `compile()`
- **Dynamic attributes**: `__getattr__`, `__setattr__`
- **Abstract methods**: `@abstractmethod` (use traits instead)
- **Introspection**: `inspect` module
- **Some magic methods**: Depending on complexity

## Migration Strategies

### Full Migration

**When to use:**
- Small projects (<1000 LOC)
- High compatibility (>90%)
- Can afford downtime

**Pros:**
- Fastest path to completion
- No dual-system complexity
- Immediate benefits

**Cons:**
- Higher risk
- Requires complete freeze
- All-or-nothing approach

### Incremental Migration

**When to use:**
- Medium projects (1K-10K LOC)
- Good compatibility (70-90%)
- Need gradual rollout

**Pros:**
- Lower risk per batch
- Can roll back easily
- Continuous deployment

**Cons:**
- Longer total timeline
- Dual-system maintenance
- Integration complexity

### Bottom-Up Migration

**When to use:**
- Projects with clear dependencies
- Need maximum safety
- Complex dependency graph

**Pros:**
- Safest approach
- Dependencies always ready
- Can run both versions in parallel

**Cons:**
- Slower to see benefits
- Requires dependency analysis
- May need temporary adapters

### Top-Down Migration

**When to use:**
- Want quick wins
- Tight deadline
- Critical path focus

**Pros:**
- Fast time to value
- User-facing features first
- Clear progress

**Cons:**
- Requires temporary shims
- Higher risk
- May break abstractions

### Critical Path Migration

**When to use:**
- Performance bottlenecks identified
- ROI-focused approach
- Resource constraints

**Pros:**
- Maximize business value
- Focus on what matters
- Flexible prioritization

**Cons:**
- May need refactoring
- Complex dependencies
- Requires good judgment

## Best Practices

### Before Assessment

1. **Update Dependencies**: Ensure Python packages are current
2. **Run Tests**: Verify existing test coverage
3. **Document Patterns**: Note any unusual code patterns
4. **Identify Owners**: Know who owns each module

### During Assessment

1. **Review Blockers**: Understand each blocker's impact
2. **Validate Metrics**: Check LOC and complexity estimates
3. **Consult Team**: Discuss feasibility with developers
4. **Plan Refactoring**: Identify blocker removal strategies

### After Assessment

1. **Prioritize Work**: Focus on high-impact blockers
2. **Prototype**: Test translation on critical modules
3. **Set Milestones**: Define clear success criteria
4. **Track Progress**: Monitor metrics during migration

## Troubleshooting

### Low Compatibility Score (<50%)

**Likely causes:**
- Heavy use of metaclasses
- Dynamic code generation
- Complex introspection

**Solutions:**
- Refactor to use composition
- Replace dynamic code with static
- Consider alternative approaches

### High Effort Estimate (>6 months)

**Likely causes:**
- Large codebase (>50K LOC)
- Many blockers
- Complex dependencies

**Solutions:**
- Split into smaller projects
- Focus on critical modules first
- Consider partial migration

### Circular Dependencies

**Symptoms:**
- Plan command fails
- "Circular dependency detected" error

**Solutions:**
- Refactor to break cycles
- Use dependency injection
- Restructure module hierarchy

## Examples

### Example 1: Simple Calculator

```bash
# Assess the sample calculator project
portalis assess --project examples/assessment-reports/sample-project --verbose

# Output:
# Translatability: 95%
# Fully Supported: 18 features
# Blockers: 0
# Estimated Effort: 2-3 days
```

### Example 2: Web Framework

```bash
# Assess a larger web application
portalis assess --project ~/projects/my-web-app --report webapp-assessment.html

# Generate migration plan
portalis plan --project ~/projects/my-web-app --strategy incremental
```

### Example 3: Data Processing Pipeline

```bash
# Check compatibility for data pipeline
portalis assess --project ~/data-pipeline --format json > pipeline-report.json

# Parse with jq
jq '.compatibility.score.overall' pipeline-report.json
```

## FAQ

**Q: How accurate are the effort estimates?**
A: Estimates are based on industry averages and code complexity. Actual time may vary ±30% based on team experience and code quality.

**Q: Can I customize the feature support matrix?**
A: Not currently, but this is planned for future releases. You can modify the source code in `core/src/assessment/feature_detector.rs`.

**Q: What if my code uses unsupported features?**
A: Review the suggested workarounds in the report. Many unsupported features can be refactored to use supported patterns.

**Q: How often should I re-assess?**
A: Re-assess after:
- Major refactoring
- Adding new features
- Before starting migration
- After each migration phase

**Q: Can I use this for other languages?**
A: Currently Python-only. Support for other languages may be added in future versions.

## Support

For issues or questions:
- GitHub Issues: https://github.com/portalis/portalis/issues
- Documentation: https://docs.portalis.dev
- Community: https://discord.gg/portalis

## Contributing

We welcome contributions! See `CONTRIBUTING.md` for guidelines.

## License

MIT License - see `LICENSE` file for details.
