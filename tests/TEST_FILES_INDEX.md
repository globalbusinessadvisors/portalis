# Test Suite File Index

## Quick Reference for All Test Files

### Core Test Modules

| File | Purpose | Lines | Tests |
|------|---------|-------|-------|
| `conftest.py` | Shared fixtures, mocks, configuration | 466 | N/A (fixtures) |
| `integration/test_nemo_cuda_integration.py` | NeMo + CUDA integration | 372 | ~20 tests |
| `integration/test_triton_nim_integration.py` | Triton + NIM integration | 534 | ~30 tests |
| `integration/test_dgx_cloud_integration.py` | DGX Cloud orchestration | 556 | ~25 tests |
| `integration/test_omniverse_wasm_integration.py` | Omniverse WASM loading | 278 | ~15 tests |
| `e2e/test_full_translation_pipeline.py` | Full E2E pipeline | 483 | ~10 tests |
| `performance/test_benchmarks.py` | Performance benchmarks | 376 | ~15 tests |
| `security/test_security_validation.py` | Security validation | 364 | ~20 tests |

**Total: 8 modules, ~135 tests, 3,429 lines**

### Configuration Files

| File | Purpose |
|------|---------|
| `pytest.ini` | Pytest configuration, markers, settings |
| `.coveragerc` | Coverage configuration and exclusions |
| `requirements-test.txt` | Test dependencies |

### Infrastructure Files

| File | Purpose |
|------|---------|
| `../docker-compose.test.yaml` | Multi-service test environment |
| `../Dockerfile.test` | Test container image |
| `../.github/workflows/test.yml` | CI/CD pipeline configuration |

### Documentation Files

| File | Purpose | Lines |
|------|---------|-------|
| `README.md` | Test suite user guide | 420 |
| `../TESTING_STRATEGY.md` | Overall testing strategy | 650 |
| `../TEST_SUITE_SUMMARY.md` | Implementation summary | ~500 |
| `TEST_FILES_INDEX.md` | This file | ~50 |

**Total Documentation: 1,620 lines**

---

**Total Test Suite: 17 files, ~5,049 lines**
