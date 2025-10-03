# Portalis NVIDIA Stack Test Suite

Comprehensive test suite for the Portalis NVIDIA technology integration stack.

## Overview

This test suite validates all 6 NVIDIA integrations:

1. **NeMo Integration** - Python to Rust translation
2. **CUDA Acceleration** - GPU-accelerated processing
3. **Triton Deployment** - Model serving infrastructure
4. **NIM Microservices** - API endpoints and orchestration
5. **DGX Cloud** - Distributed workload management
6. **Omniverse Integration** - WASM loading and execution

## Test Structure

```
tests/
├── conftest.py                           # Shared fixtures and configuration
├── pytest.ini                            # Pytest configuration
├── requirements-test.txt                 # Test dependencies
├── .coveragerc                          # Coverage configuration
│
├── integration/                         # Integration tests
│   ├── test_nemo_cuda_integration.py   # NeMo + CUDA pipeline
│   ├── test_triton_nim_integration.py  # Triton + NIM integration
│   ├── test_dgx_cloud_integration.py   # DGX Cloud orchestration
│   └── test_omniverse_wasm_integration.py  # Omniverse WASM
│
├── e2e/                                # End-to-end tests
│   └── test_full_translation_pipeline.py
│
├── performance/                        # Performance benchmarks
│   └── test_benchmarks.py
│
├── security/                           # Security validation
│   └── test_security_validation.py
│
└── fixtures/                           # Test data and mocks
    ├── data/
    └── mocks/
```

## Quick Start

### Prerequisites

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# For GPU tests (optional)
pip install cupy-cuda12x nvidia-cuda-runtime-cu12
```

### Running Tests

#### All Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

#### By Category
```bash
# Unit tests only (fast)
pytest tests/ -m unit -v

# Integration tests
pytest tests/ -m integration -v

# End-to-end tests
pytest tests/ -m e2e -v

# Performance benchmarks
pytest tests/ -m benchmark -v

# Security tests
pytest tests/ -m security -v
```

#### By Component
```bash
# NeMo tests
pytest tests/ -m nemo -v

# CUDA tests
pytest tests/ -m cuda -v

# Triton tests
pytest tests/ -m triton -v

# NIM tests
pytest tests/ -m nim -v

# DGX Cloud tests
pytest tests/ -m dgx -v

# Omniverse tests
pytest tests/ -m omniverse -v
```

#### GPU Tests
```bash
# Run GPU-specific tests (requires CUDA)
ENABLE_GPU_TESTS=true pytest tests/ -m gpu -v

# Run CUDA tests
ENABLE_GPU_TESTS=true pytest tests/ -m cuda -v
```

### Parallel Execution

```bash
# Run tests in parallel (4 workers)
pytest tests/ -n 4 -v

# Auto-detect number of CPUs
pytest tests/ -n auto -v
```

## Test Markers

| Marker | Description | Speed | GPU Required |
|--------|-------------|-------|--------------|
| `unit` | Unit tests | Fast (< 1s) | No |
| `integration` | Integration tests | Medium (1-30s) | No |
| `e2e` | End-to-end tests | Slow (30s-5m) | No |
| `gpu` | GPU tests | Varies | Yes |
| `cuda` | CUDA tests | Varies | Yes |
| `slow` | Slow tests | > 5s | No |
| `benchmark` | Performance tests | Varies | No |
| `security` | Security tests | Fast | No |
| `smoke` | Quick smoke tests | Fast | No |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_URL` | `localhost:8000` | Triton server URL |
| `NIM_API_URL` | `http://localhost:8000` | NIM API URL |
| `DGX_CLOUD_URL` | `http://localhost:8080` | DGX Cloud URL |
| `TEST_TIMEOUT` | `300` | Test timeout in seconds |
| `ENABLE_GPU_TESTS` | `false` | Enable GPU tests |
| `ENABLE_NETWORK_TESTS` | `true` | Enable network tests |
| `ENABLE_SLOW_TESTS` | `true` | Enable slow tests |
| `MOCK_MODE` | `true` | Use mocks instead of real services |

## Docker Testing

### Using Docker Compose

```bash
# Start all services and run tests
docker-compose -f docker-compose.test.yaml up --abort-on-container-exit

# Run specific test suite
docker-compose -f docker-compose.test.yaml run pytest pytest tests/integration/ -v

# Cleanup
docker-compose -f docker-compose.test.yaml down -v
```

### Individual Containers

```bash
# Build test image
docker build -f Dockerfile.test -t portalis-test .

# Run tests
docker run --rm -v $(pwd):/workspace portalis-test pytest tests/ -v
```

## CI/CD Pipeline

Tests run automatically on:
- **Push to main/develop** - Unit + Integration tests
- **Pull requests** - Full test suite
- **Nightly** - Full test suite including GPU tests
- **Manual trigger** - Configurable test selection

### GitHub Actions Workflow

The workflow includes:
1. **Unit Tests** - Fast tests, no external dependencies
2. **Integration Tests** - Tests with local services
3. **Security Tests** - Bandit, Safety, security validation
4. **Performance Tests** - Benchmarks and regression detection
5. **GPU Tests** - CUDA/GPU tests on self-hosted runners
6. **E2E Tests** - Full stack testing with Docker
7. **Code Quality** - Black, Ruff, MyPy checks

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Translation P95 Latency | < 500ms | TBD |
| Translation Throughput | > 10 req/s | TBD |
| E2E Pipeline Time | < 5 min | TBD |
| Omniverse FPS | > 30 FPS | TBD |
| GPU Memory Usage | < 8GB | TBD |
| Code Coverage | > 80% | TBD |

## Coverage Reports

### Generate Coverage Report

```bash
# HTML report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Terminal report
pytest tests/ --cov=. --cov-report=term-missing

# XML report (for CI)
pytest tests/ --cov=. --cov-report=xml
```

### Coverage Targets

- **Overall**: > 80%
- **NeMo Integration**: > 85%
- **CUDA Acceleration**: > 75%
- **Triton Deployment**: > 80%
- **NIM Microservices**: > 90%
- **DGX Cloud**: > 80%
- **Omniverse**: > 75%

## Debugging Tests

### Run Single Test

```bash
pytest tests/integration/test_nemo_cuda_integration.py::TestNeMoCUDAIntegration::test_translate_and_accelerate -v
```

### Enable Debug Logging

```bash
pytest tests/ -v --log-cli-level=DEBUG
```

### Keep Failed Test Artifacts

```bash
pytest tests/ -v --keep-failed
```

### Interactive Debugging

```bash
# Drop into debugger on failure
pytest tests/ -v --pdb

# Drop into debugger on first failure
pytest tests/ -v -x --pdb
```

## Common Issues

### CUDA Out of Memory

```bash
# Reduce batch size or skip GPU tests
ENABLE_GPU_TESTS=false pytest tests/ -v
```

### Services Not Available

```bash
# Run in mock mode
MOCK_MODE=true pytest tests/ -v
```

### Slow Tests

```bash
# Skip slow tests
pytest tests/ -m "not slow" -v

# Run only smoke tests
pytest tests/ -m smoke -v
```

## Writing New Tests

### Test Template

```python
import pytest

@pytest.mark.integration
@pytest.mark.nemo
class TestNewFeature:
    """Test new NeMo feature."""

    @pytest.fixture
    def setup(self):
        """Setup test environment."""
        # Setup code
        yield
        # Cleanup code

    def test_basic_functionality(self, setup):
        """Test basic functionality."""
        # Arrange
        input_data = "test"

        # Act
        result = process(input_data)

        # Assert
        assert result is not None
        assert result.success is True
```

### Best Practices

1. **Use fixtures** for shared setup/teardown
2. **Mark tests appropriately** with pytest markers
3. **Include docstrings** explaining what is tested
4. **Test edge cases** and error conditions
5. **Use parametrize** for similar test cases
6. **Keep tests isolated** - no dependencies between tests
7. **Clean up resources** in fixtures and teardowns
8. **Use appropriate timeouts** for async tests

## Test Data

Test data is stored in:
- `tests/fixtures/data/` - Static test data
- `tests/fixtures/mocks/` - Mock responses
- Generated dynamically using `python_code_generator` fixture

## Continuous Improvement

### Performance Regression

Tests automatically detect performance regressions:
- Baseline metrics stored in `test_benchmarks.py`
- 20% threshold for alerts
- Nightly comparison against baseline

### Coverage Tracking

Coverage is tracked per:
- Component
- Feature
- Test category

### Test Health Metrics

Monitored metrics:
- Test execution time
- Flaky test rate
- Test failure patterns
- Coverage trends

## Support

For test-related issues:
1. Check this README
2. Review test logs in `test-reports/`
3. Check CI/CD pipeline logs
4. Contact the QA team

## Contributing

When adding new features:
1. Add corresponding tests
2. Update test documentation
3. Ensure coverage targets are met
4. Run full test suite locally
5. Verify CI/CD passes

---

**Last Updated**: 2025-10-03
**Test Suite Version**: 1.0.0
**Maintainer**: QA Engineering Team
