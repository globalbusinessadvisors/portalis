# Portalis NeMo Integration

GPU-accelerated Python to Rust translation engine powered by NVIDIA NeMo framework.

## Overview

This module provides the core translation infrastructure for the Portalis platform, leveraging:
- **NeMo Language Models** for intelligent code translation
- **CUDA Acceleration** for batch processing and embedding generation
- **Semantic Understanding** for preserving Python semantics in Rust
- **Type System Mapping** for accurate Python→Rust type translations

## Architecture

```
nemo-integration/
├── src/
│   ├── translation/         # Core translation engine
│   │   ├── __init__.py
│   │   ├── nemo_service.py  # NeMo model wrapper
│   │   ├── translator.py    # Main translation logic
│   │   └── templates.py     # Translation templates
│   ├── validation/          # Translation validation
│   │   ├── __init__.py
│   │   ├── validator.py     # Code validation
│   │   └── semantic_checker.py
│   ├── mapping/             # Type system mapping
│   │   ├── __init__.py
│   │   ├── type_mapper.py   # Python→Rust type mapping
│   │   └── error_mapper.py  # Exception handling
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── ast_utils.py
│       └── cuda_utils.py
├── config/                  # Model configurations
├── tests/                   # Unit and integration tests
└── models/                  # Model artifacts
```

## Features

### Translation Engine
- AST-based Python analysis
- NeMo-powered code generation
- Template-based translation patterns
- GPU-accelerated batch processing

### Type System
- Comprehensive Python→Rust type mapping
- Generic type handling (Optional, Union, etc.)
- Custom type definitions
- Stdlib type mappings

### Validation
- Syntactic validation of generated Rust code
- Semantic equivalence checking
- Contract preservation
- Error handling verification

### CUDA Acceleration
- Parallel AST parsing
- GPU-based embedding generation
- k-NN similarity search for examples
- Batch inference optimization

## Installation

```bash
# Install with CUDA support
pip install -e ".[cuda]"

# Development installation
pip install -e ".[dev,cuda]"
```

## Quick Start

```python
from portalis_nemo.translation import NeMoTranslator
from portalis_nemo.config import TranslationConfig

# Initialize translator
config = TranslationConfig(
    model_name="portalis/python-rust-translator",
    gpu_enabled=True,
    batch_size=32
)
translator = NeMoTranslator(config)

# Translate Python code
python_code = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

result = translator.translate(python_code)
print(result.rust_code)
```

## Configuration

See `config/` directory for model configuration examples:
- `translation_model.yaml` - NeMo translation model config
- `embedding_model.yaml` - Embedding generation config
- `type_mappings.yaml` - Type system mappings

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_translation/
```

## Performance

Expected performance on NVIDIA A100:
- **Simple functions**: ~50ms per function
- **Complex classes**: ~200ms per class
- **Batch processing**: 100+ functions/second
- **GPU speedup**: 10-50x over CPU-only

## Requirements

- Python 3.10+
- CUDA 12.0+
- NVIDIA GPU (compute capability 7.0+)
- 16GB+ system RAM
- 8GB+ GPU memory

## License

MIT License - see LICENSE file for details
