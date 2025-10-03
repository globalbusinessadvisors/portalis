# NeMo Integration Implementation Guide

## Overview

This document provides a comprehensive guide to the NeMo language model integration for the Portalis Python→Rust translation platform.

## Architecture

### Component Structure

```
nemo-integration/
├── src/
│   ├── translation/          # Core translation engine
│   │   ├── nemo_service.py   # NeMo model wrapper with GPU acceleration
│   │   ├── translator.py     # High-level translation orchestration
│   │   └── templates.py      # Translation pattern templates
│   ├── mapping/              # Type system mapping
│   │   ├── type_mapper.py    # Python→Rust type translation
│   │   └── error_mapper.py   # Exception→Result mapping
│   ├── validation/           # Code validation
│   │   ├── validator.py      # Syntax and semantic validation
│   │   └── semantic_checker.py  # Semantic equivalence checking
│   └── utils/                # Utility functions
│       ├── ast_utils.py      # AST parsing utilities
│       └── cuda_utils.py     # CUDA/GPU utilities
├── config/                   # Configuration files
│   ├── translation_model.yaml    # NeMo model configuration
│   └── type_mappings.yaml        # Type mapping definitions
├── tests/                    # Unit and integration tests
└── models/                   # Model artifacts (gitignored)
```

## Key Components

### 1. NeMo Service (`src/translation/nemo_service.py`)

**Purpose**: Wrapper around NVIDIA NeMo models for code translation inference.

**Key Features**:
- Model loading and initialization
- GPU-accelerated batch inference
- Embedding generation for code similarity
- Automatic retry logic with exponential backoff
- Memory management and cleanup
- CPU fallback for testing

**Usage Example**:
```python
from src.translation.nemo_service import NeMoService, InferenceConfig

# Configure inference
config = InferenceConfig(
    max_length=512,
    temperature=0.2,
    batch_size=32,
    use_gpu=True
)

# Initialize service
service = NeMoService(
    model_path="models/python-rust-translator.nemo",
    config=config
)

# Translate code
with service:
    result = service.translate_code(python_code, context)
    print(result.rust_code)
```

**Key Methods**:
- `initialize()` - Load NeMo model
- `translate_code(python_code, context)` - Single translation
- `batch_translate(codes, contexts)` - Batch translation
- `generate_embeddings(snippets)` - Code embedding generation
- `cleanup()` - Resource cleanup

### 2. Type Mapper (`src/mapping/type_mapper.py`)

**Purpose**: Comprehensive Python to Rust type system mapping.

**Key Features**:
- Primitive type mappings (int→i64, str→String, etc.)
- Collection mappings (list→Vec, dict→HashMap)
- Generic type handling (Optional→Option, Union→enum)
- Stdlib type equivalents (pathlib.Path→PathBuf)
- Custom type registration
- Import collection

**Type Categories**:
1. **Primitives**: int, float, str, bool, bytes, None
2. **Collections**: list, dict, set, tuple, frozenset
3. **Generics**: Optional, Union, List, Dict, Callable, Iterator
4. **Stdlib**: pathlib, datetime, re, io modules
5. **Custom**: User-defined types

**Usage Example**:
```python
from src.mapping.type_mapper import TypeMapper

mapper = TypeMapper()

# Map simple types
rust_type = mapper.map_annotation("int")  # → i64

# Map generic types
rust_type = mapper.map_annotation("List[int]")  # → Vec<i64>
rust_type = mapper.map_annotation("Dict[str, int]")  # → HashMap<String, i64>
rust_type = mapper.map_annotation("Optional[str]")  # → Option<String>

# Map function signatures
signature = mapper.map_function_signature(
    param_types=["int", "str", "Optional[bool]"],
    return_type="List[int]"
)
```

### 3. Error Mapper (`src/mapping/error_mapper.py`)

**Purpose**: Map Python exceptions to Rust error handling patterns.

**Key Features**:
- Standard exception mappings (ValueError, TypeError, etc.)
- Custom exception registration
- Error enum generation
- Result type wrapping
- Error handling strategy selection

**Strategies**:
1. **RESULT**: Return `Result<T, E>` (default for most exceptions)
2. **PANIC**: Use `panic!` for unrecoverable errors
3. **OPTION**: Return `Option<T>` for missing values

**Usage Example**:
```python
from src.mapping.error_mapper import ErrorMapper, ErrorHandlingStrategy

mapper = ErrorMapper()

# Get exception mapping
mapping = mapper.get_mapping("ValueError")
print(mapping.rust_error_variant)  # → "ValueError"

# Generate error enum
exceptions = ["ValueError", "TypeError", "KeyError"]
enum_code = mapper.generate_error_enum(exceptions)

# Translate raise statement
rust_code = mapper.translate_raise_statement(
    "ValueError",
    "Invalid input"
)
# → return Err(Error::ValueError("Invalid input".to_string()));
```

### 4. Translation Validator (`src/validation/validator.py`)

**Purpose**: Validate generated Rust code for correctness and quality.

**Validation Levels**:
1. **SYNTAX_ONLY**: Basic syntax checks
2. **SEMANTIC**: Semantic correctness
3. **FULL**: Complete quality analysis

**Checks Performed**:
- **Syntax**: Balanced braces, function syntax, type syntax
- **Semantic**: Error handling, ownership patterns, unsafe blocks
- **Quality**: Code formatting, naming conventions, documentation

**Usage Example**:
```python
from src.validation.validator import TranslationValidator, ValidationLevel

validator = TranslationValidator(level=ValidationLevel.FULL)

result = validator.validate(rust_code, python_code)

if result.is_valid():
    print("Validation passed!")
else:
    for error in result.get_errors():
        print(f"Error: {error.message}")
    for warning in result.get_warnings():
        print(f"Warning: {warning.message}")

print(f"Metrics: {result.metrics}")
```

### 5. Main Translator (`src/translation/translator.py`)

**Purpose**: High-level orchestration of the translation process.

**Key Features**:
- Function translation
- Class translation
- Module translation
- Batch processing
- Context enhancement with type hints
- Post-processing of generated code
- Statistics tracking

**Usage Example**:
```python
from src.translation.translator import NeMoTranslator, TranslationConfig

# Configure translator
config = TranslationConfig(
    model_path="models/translator.nemo",
    gpu_enabled=True,
    batch_size=32,
    validate_output=True
)

# Initialize translator
with NeMoTranslator(config) as translator:
    # Translate function
    python_code = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    result = translator.translate_function(python_code)
    print(result.rust_code)
    print(f"Confidence: {result.confidence}")
    print(f"Imports: {result.imports}")

    # Batch translate
    codes = [code1, code2, code3]
    results = translator.batch_translate(codes)

    # Get statistics
    stats = translator.get_statistics()
    print(f"Success rate: {stats['success_rate']}")
```

## Configuration

### Model Configuration (`config/translation_model.yaml`)

Key parameters:
- **Model Architecture**: Megatron-GPT based
- **Hidden Size**: 2048 (adjustable)
- **Max Length**: 512 tokens
- **Temperature**: 0.2 (low for deterministic code)
- **Batch Size**: 32
- **GPU Settings**: CUDA device, tensor parallelism

### Type Mappings (`config/type_mappings.yaml`)

Defines comprehensive type mappings:
- Primitive types
- Collection types
- Generic types
- Standard library types
- NumPy types
- Custom type extensions

## Testing Strategy

### Test Structure

```
tests/
├── test_nemo_service.py      # NeMo service tests
├── test_type_mapper.py        # Type mapping tests
├── test_error_mapper.py       # Error mapping tests
├── test_validator.py          # Validation tests
├── test_translator.py         # End-to-end translation tests
└── test_integration.py        # Integration tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_nemo_service.py

# Run specific test
pytest tests/test_nemo_service.py::TestNeMoService::test_translate_code

# Run with verbose output
pytest -v

# Run with GPU (if available)
CUDA_VISIBLE_DEVICES=0 pytest tests/
```

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Property Tests**: Test invariants with random inputs
4. **Performance Tests**: Benchmark translation speed

## GPU Acceleration

### CUDA Integration Points

1. **Batch Inference**: Parallel translation of multiple functions
2. **Embedding Generation**: GPU-accelerated code embeddings
3. **Similarity Search**: k-NN search for translation examples
4. **Memory Management**: Efficient GPU memory pooling

### GPU Memory Optimization

```python
from src.utils.cuda_utils import (
    check_cuda_available,
    get_gpu_memory_info,
    optimize_gpu_memory
)

# Check CUDA availability
if check_cuda_available():
    # Get memory info
    mem_info = get_gpu_memory_info()
    print(f"GPU Memory: {mem_info['free_mb']:.2f} MB free")

    # Optimize memory usage
    optimize_gpu_memory()
```

## Error Handling

### Retry Logic

The NeMo service includes automatic retry with exponential backoff:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def translate_code(self, python_code, context):
    # Translation logic with automatic retry
    pass
```

### CPU Fallback

GPU operations automatically fall back to CPU if CUDA is unavailable:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Performance Optimization

### Caching

- Translation results cached by code hash
- Embeddings cached in GPU memory
- Type mappings cached for reuse

### Batching

```python
# Efficient batch translation
results = translator.batch_translate(
    python_codes=[code1, code2, code3, ...],
    contexts=[ctx1, ctx2, ctx3, ...]
)
```

### Memory Management

- Automatic GPU cache clearing
- Memory pooling for repeated operations
- Streaming for large codebases

## Production Deployment

### Docker Deployment

```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app
COPY . /app

RUN pip install -e ".[cuda]"

CMD ["python", "-m", "src.translation.translator"]
```

### Triton Integration

The model can be deployed on Triton Inference Server:

```yaml
deployment:
  backend: "triton"
  max_batch_size: 32
  dynamic_batching: true
```

## Monitoring and Logging

### Logging Configuration

```python
from loguru import logger

logger.add(
    "logs/translation_{time}.log",
    rotation="100 MB",
    retention="7 days",
    level="INFO"
)
```

### Metrics

Track translation metrics:
- Success rate
- Processing time
- Confidence scores
- GPU utilization

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Clear GPU cache regularly

2. **Low Confidence Scores**
   - Provide more context
   - Use few-shot examples
   - Fine-tune model on domain-specific data

3. **Slow Performance**
   - Enable GPU acceleration
   - Increase batch size
   - Use model quantization

## Next Steps

1. **Model Training**: Train NeMo model on Python-Rust pairs
2. **Fine-tuning**: Domain-specific fine-tuning
3. **Optimization**: Quantization and pruning
4. **Deployment**: Production deployment on Triton
5. **Monitoring**: Set up metrics and alerting

## References

- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/)
- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Rust Language](https://www.rust-lang.org/)
- [libCST Documentation](https://libcst.readthedocs.io/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/portalis/portalis/issues
- Documentation: https://portalis.readthedocs.io
- Email: support@portalis.ai
