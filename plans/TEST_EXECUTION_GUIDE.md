# Test Execution Guide
## Portalis NVIDIA Stack Testing

**Version**: 1.0
**Last Updated**: 2025-10-03

---

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install pytest pytest-asyncio pytest-cov pytest-timeout pytest-html pytest-xdist
pip install httpx torch numpy psutil

# Optional: For GPU tests
# Ensure CUDA toolkit is installed

# Optional: For integration tests
# Start required services (Triton, NIM, etc.)
```

### Run All Tests

```bash
# From project root
cd /workspace/portalis
pytest tests/ -v
```

---

## Test Execution by Category

### Unit Tests (Fast - Recommended for Development)

```bash
# Run all unit tests
pytest -m unit -v

# Run specific unit test file
pytest tests/unit/test_translation_routes.py -v

# Run with coverage
pytest -m unit --cov=. --cov-report=html
```

**Expected Duration**: 10-30 seconds
**Expected Pass Rate**: 95-100%

---

### Integration Tests

```bash
# Run all integration tests
pytest -m integration -v

# Skip network-dependent tests
pytest -m integration -v --skip-network

# Run specific integration
pytest tests/integration/test_nemo_cuda_integration.py -v
```

**Expected Duration**: 2-5 minutes
**Expected Pass Rate**: 85-95%
**Note**: Some tests may be skipped if services unavailable

---

### End-to-End Tests

```bash
# Run all E2E tests
pytest -m e2e -v

# Run with longer timeout
pytest -m e2e -v --timeout=600
```

**Expected Duration**: 5-10 minutes
**Expected Pass Rate**: 80-90%
**Note**: Requires all services running

---

### Performance/Benchmark Tests

```bash
# Run all benchmark tests
pytest -m benchmark -v

# Run specific benchmark
pytest tests/performance/test_benchmarks.py::TestTranslationPerformance -v

# Save benchmark results
pytest -m benchmark --benchmark-autosave
```

**Expected Duration**: 3-5 minutes
**Expected Pass Rate**: 90-100%

---

### GPU/CUDA Tests

```bash
# Enable GPU tests
ENABLE_GPU_TESTS=true pytest -m cuda -v

# Run only if CUDA available
pytest -m cuda -v --skip-if-no-cuda
```

**Expected Duration**: 2-4 minutes
**Expected Pass Rate**: 90-100%
**Requirement**: CUDA-capable GPU

---

### Security Tests

```bash
# Run all security tests
pytest -m security -v

# Run with verbose output
pytest tests/security/test_security_validation.py -vv
```

**Expected Duration**: 1-2 minutes
**Expected Pass Rate**: 100%

---

## Advanced Test Execution

### Parallel Execution

```bash
# Run tests in parallel (auto-detect CPU cores)
pytest -n auto -v

# Specify number of workers
pytest -n 4 -v

# Parallel with coverage (requires pytest-cov)
pytest -n auto --cov=. --cov-report=html
```

**Speedup**: 2-4x faster on multi-core systems

---

### Test Selection

```bash
# Run tests matching pattern
pytest -k "translation" -v

# Run tests NOT matching pattern
pytest -k "not slow" -v

# Combine markers
pytest -m "integration and not slow" -v

# Run only failed tests from last run
pytest --lf -v

# Run failed first, then others
pytest --ff -v
```

---

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=. --cov-report=html -v
# Open htmlcov/index.html

# Generate XML coverage (for CI/CD)
pytest --cov=. --cov-report=xml -v

# Terminal coverage with missing lines
pytest --cov=. --cov-report=term-missing -v

# Set minimum coverage threshold
pytest --cov=. --cov-fail-under=80 -v
```

---

### Debugging Tests

```bash
# Stop on first failure
pytest -x -v

# Drop into debugger on failure
pytest --pdb -v

# Show local variables on failure
pytest -l -v

# Increase verbosity
pytest -vv

# Show print statements
pytest -s -v

# Capture output per test
pytest --capture=no -v
```

---

## Environment Configuration

### Environment Variables

```bash
# Enable GPU tests
export ENABLE_GPU_TESTS=true

# Enable network-dependent tests
export ENABLE_NETWORK_TESTS=true

# Enable slow tests
export ENABLE_SLOW_TESTS=true

# Service endpoints
export TRITON_URL="localhost:8000"
export NIM_API_URL="http://localhost:8000"
export DGX_CLOUD_URL="http://localhost:8080"

# Test timeout (seconds)
export TEST_TIMEOUT=300

# Enable mock mode (no real services)
export MOCK_MODE=true
```

### Configuration File

Create `.env` file in project root:

```bash
# .env
ENABLE_GPU_TESTS=false
ENABLE_NETWORK_TESTS=true
ENABLE_SLOW_TESTS=false
TRITON_URL=localhost:8000
NIM_API_URL=http://localhost:8000
MOCK_MODE=true
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements-test.txt

    - name: Run unit tests
      run: |
        pytest -m unit --cov=. --cov-report=xml

    - name: Run integration tests
      run: |
        pytest -m integration -v

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Test Reporting

### HTML Report

```bash
# Generate HTML test report
pytest --html=test-reports/report.html --self-contained-html -v

# Open report
open test-reports/report.html
```

### JUnit XML (for CI)

```bash
# Generate JUnit XML
pytest --junitxml=test-reports/junit.xml -v
```

### Custom Reporting

```bash
# Show slowest tests
pytest --durations=10 -v

# Show test summary
pytest -ra -v

# Show all outcomes
pytest -r A -v
```

---

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

```bash
# Ensure you're in project root
cd /workspace/portalis

# Install missing dependencies
pip install -r requirements-test.txt

# Add project to Python path
export PYTHONPATH=/workspace/portalis:$PYTHONPATH
```

#### 2. CUDA Tests Failing

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Disable GPU tests if no CUDA
export ENABLE_GPU_TESTS=false
pytest -v
```

#### 3. Service Connection Failures

```bash
# Use mock mode
export MOCK_MODE=true
pytest -v

# Skip network tests
pytest -m "not requires_network" -v

# Check service availability
curl http://localhost:8000/health
```

#### 4. Slow Tests Timeout

```bash
# Increase timeout
pytest --timeout=600 -v

# Skip slow tests
pytest -m "not slow" -v

# Disable timeout
pytest --timeout=0 -v
```

#### 5. Permission Errors

```bash
# Create test directories
mkdir -p test-reports htmlcov

# Set permissions
chmod 755 test-reports htmlcov
```

---

## Best Practices

### During Development

```bash
# Run related tests only
pytest tests/unit/test_translation_routes.py -v

# Watch for file changes (requires pytest-watch)
ptw -- -v

# Run last failed tests
pytest --lf -v
```

### Before Committing

```bash
# Run all unit tests
pytest -m unit -v

# Check coverage
pytest -m unit --cov=. --cov-report=term-missing

# Ensure no failures
pytest -m unit -x
```

### Before Pull Request

```bash
# Run full test suite (except GPU/slow)
pytest -m "not (gpu or slow)" -v

# Generate coverage report
pytest --cov=. --cov-report=html

# Check coverage threshold
pytest --cov=. --cov-fail-under=80
```

### In CI/CD

```bash
# Fast feedback
pytest -m unit -v

# Integration validation
pytest -m integration -v

# Full validation (nightly)
pytest -v --cov=. --cov-report=xml
```

---

## Test Organization Reference

### Test Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── pytest.ini                  # Pytest configuration
├── unit/                       # Unit tests
│   ├── test_translation_routes.py
│   ├── test_health_routes.py
│   └── test_nemo_service_collaboration.py
├── integration/                # Integration tests
│   ├── test_nemo_cuda_integration.py
│   ├── test_triton_nim_integration.py
│   ├── test_dgx_cloud_integration.py
│   └── test_omniverse_wasm_integration.py
├── e2e/                        # End-to-end tests
│   └── test_full_translation_pipeline.py
├── performance/                # Performance tests
│   └── test_benchmarks.py
├── security/                   # Security tests
│   └── test_security_validation.py
└── acceptance/                 # Acceptance tests
    └── test_translation_workflow.py
```

### Test Markers

Available markers (from `pytest.ini`):

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.gpu` - GPU-dependent tests
- `@pytest.mark.cuda` - CUDA-specific tests
- `@pytest.mark.slow` - Slow-running tests (>5s)
- `@pytest.mark.security` - Security validation
- `@pytest.mark.smoke` - Quick smoke tests
- `@pytest.mark.nemo` - NeMo-specific tests
- `@pytest.mark.triton` - Triton-specific tests
- `@pytest.mark.nim` - NIM microservice tests
- `@pytest.mark.dgx` - DGX Cloud tests
- `@pytest.mark.omniverse` - Omniverse tests
- `@pytest.mark.wasm` - WebAssembly tests
- `@pytest.mark.requires_network` - Requires network
- `@pytest.mark.requires_docker` - Requires Docker

---

## Performance Tuning

### Speed Up Tests

```bash
# 1. Run in parallel
pytest -n auto

# 2. Skip slow tests
pytest -m "not slow"

# 3. Use mock mode
export MOCK_MODE=true
pytest -v

# 4. Disable coverage (faster)
pytest -v  # without --cov

# 5. Stop on first failure
pytest -x

# 6. Run subset
pytest tests/unit/ -v
```

### Optimize Coverage Collection

```bash
# Collect coverage only for source code
pytest --cov=src --cov=nemo-integration --cov=nim-microservices

# Skip coverage for tests
pytest --cov=. --cov-config=.coveragerc

# Parallel with coverage
pytest -n auto --cov=. --cov-context=test
```

---

## Continuous Monitoring

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run unit tests before commit

echo "Running unit tests..."
pytest -m unit -v

if [ $? -ne 0 ]; then
    echo "Unit tests failed. Commit aborted."
    exit 1
fi

echo "All tests passed!"
exit 0
```

### Watch Mode

```bash
# Install pytest-watch
pip install pytest-watch

# Watch for changes and run tests
ptw -- -m unit -v

# Watch specific directory
ptw tests/unit/ -- -v
```

---

## Metrics & Reporting

### Generate Test Metrics

```bash
# Test execution time
pytest --durations=0 -v > test_durations.txt

# Coverage by component
pytest --cov=src --cov=nemo-integration --cov-report=term

# Test count by category
pytest --collect-only -q | wc -l
```

### Track Test Health

```bash
# Failed test count
pytest --tb=no -q | grep failed

# Pass rate
pytest --tb=no -q | grep passed

# Skipped test count
pytest --tb=no -q | grep skipped
```

---

## Support & Resources

### Documentation
- Pytest: https://docs.pytest.org
- pytest-asyncio: https://pytest-asyncio.readthedocs.io
- pytest-cov: https://pytest-cov.readthedocs.io

### Internal Resources
- QA Validation Report: `/workspace/portalis/QA_VALIDATION_REPORT.md`
- Test Configuration: `/workspace/portalis/tests/pytest.ini`
- Shared Fixtures: `/workspace/portalis/tests/conftest.py`

### Getting Help

For test-related questions:
1. Check this guide
2. Review test examples in `tests/unit/`
3. Consult QA Validation Report
4. Contact QA team

---

## Appendix: Sample Test Commands

### Development Workflow

```bash
# 1. Write test
vim tests/unit/test_new_feature.py

# 2. Run test
pytest tests/unit/test_new_feature.py -v

# 3. Debug if needed
pytest tests/unit/test_new_feature.py -vv -s --pdb

# 4. Check coverage
pytest tests/unit/test_new_feature.py --cov=src.new_feature

# 5. Run full unit suite
pytest -m unit -v

# 6. Commit
git add tests/unit/test_new_feature.py
git commit -m "Add tests for new feature"
```

### CI/CD Pipeline

```bash
# Stage 1: Fast tests (< 1 min)
pytest -m unit -v --tb=short

# Stage 2: Integration (< 5 min)
pytest -m "unit or integration" -v

# Stage 3: Full suite (< 15 min)
pytest -m "not slow" -v --cov=.

# Stage 4: Nightly (< 30 min)
pytest -v --cov=. --cov-report=html --html=report.html
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-03
**Maintained By**: QA Team
