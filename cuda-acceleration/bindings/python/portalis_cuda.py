"""
Portalis CUDA Acceleration - Python Bindings

This module provides Python bindings for GPU-accelerated parsing,
embedding generation, and verification tasks.
"""

import ctypes
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import os

# Load the CUDA library
_lib_path = Path(__file__).parent.parent.parent / "build" / "libportalis_cuda_kernels.so"

if not _lib_path.exists():
    # Try CPU fallback
    _lib_path = Path(__file__).parent.parent.parent / "build" / "libportalis_cpu_fallback.so"
    _use_cpu_fallback = True
else:
    _use_cpu_fallback = False

try:
    _lib = ctypes.CDLL(str(_lib_path))
except OSError as e:
    raise ImportError(f"Failed to load Portalis CUDA library: {e}")

# Define C structures
class ASTNode(ctypes.Structure):
    _fields_ = [
        ("node_type", ctypes.c_uint32),
        ("parent_idx", ctypes.c_uint32),
        ("first_child", ctypes.c_uint32),
        ("next_sibling", ctypes.c_uint32),
        ("token_start", ctypes.c_uint32),
        ("token_end", ctypes.c_uint32),
        ("line_number", ctypes.c_uint32),
        ("col_number", ctypes.c_uint32),
        ("confidence", ctypes.c_float),
        ("metadata_idx", ctypes.c_uint32),
    ]

class ParserConfig(ctypes.Structure):
    _fields_ = [
        ("max_nodes", ctypes.c_uint32),
        ("max_tokens", ctypes.c_uint32),
        ("max_depth", ctypes.c_uint32),
        ("batch_size", ctypes.c_uint32),
        ("enable_async", ctypes.c_bool),
        ("collect_metrics", ctypes.c_bool),
    ]

class ParserMetrics(ctypes.Structure):
    _fields_ = [
        ("tokenization_time_ms", ctypes.c_float),
        ("parsing_time_ms", ctypes.c_float),
        ("total_time_ms", ctypes.c_float),
        ("nodes_created", ctypes.c_uint32),
        ("tokens_processed", ctypes.c_uint32),
        ("gpu_utilization", ctypes.c_float),
    ]

class EmbeddingConfig(ctypes.Structure):
    _fields_ = [
        ("vocab_size", ctypes.c_uint32),
        ("embedding_dim", ctypes.c_uint32),
        ("max_sequence_length", ctypes.c_uint32),
        ("batch_size", ctypes.c_uint32),
        ("dropout_rate", ctypes.c_float),
        ("use_fp16", ctypes.c_bool),
    ]

class EmbeddingMetrics(ctypes.Structure):
    _fields_ = [
        ("encoding_time_ms", ctypes.c_float),
        ("similarity_time_ms", ctypes.c_float),
        ("total_time_ms", ctypes.c_float),
        ("sequences_processed", ctypes.c_uint32),
        ("throughput_seq_per_sec", ctypes.c_float),
        ("gpu_memory_used_mb", ctypes.c_float),
    ]

class SimilarityResult(ctypes.Structure):
    _fields_ = [
        ("query_idx", ctypes.c_uint32),
        ("match_idx", ctypes.c_uint32),
        ("similarity_score", ctypes.c_float),
        ("confidence", ctypes.c_float),
    ]

# Define function prototypes
_lib.initializeASTParser.argtypes = [ctypes.POINTER(ParserConfig)]
_lib.initializeASTParser.restype = ctypes.c_int

_lib.parseSource.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.POINTER(ASTNode)),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ParserMetrics)
]
_lib.parseSource.restype = ctypes.c_int

_lib.cleanupASTParser.argtypes = []
_lib.cleanupASTParser.restype = ctypes.c_int

_lib.initializeEmbeddingGenerator.argtypes = [
    ctypes.POINTER(EmbeddingConfig),
    ctypes.POINTER(ctypes.c_float)
]
_lib.initializeEmbeddingGenerator.restype = ctypes.c_int

_lib.generateEmbeddings.argtypes = [
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(EmbeddingMetrics)
]
_lib.generateEmbeddings.restype = ctypes.c_int

_lib.computeSimilarity.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint32,
    ctypes.POINTER(SimilarityResult),
    ctypes.c_uint32,
    ctypes.POINTER(EmbeddingMetrics)
]
_lib.computeSimilarity.restype = ctypes.c_int

_lib.cleanupEmbeddingGenerator.argtypes = []
_lib.cleanupEmbeddingGenerator.restype = ctypes.c_int


# Python wrapper classes

@dataclass
class ASTParseResult:
    """Result of AST parsing operation"""
    nodes: List[Dict]
    metrics: Dict[str, float]
    success: bool
    error: Optional[str] = None


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embeddings: np.ndarray
    metrics: Dict[str, float]
    success: bool
    error: Optional[str] = None


class CUDAASTParser:
    """GPU-accelerated Python AST parser"""

    def __init__(self, max_nodes: int = 100000, max_tokens: int = 500000):
        self.config = ParserConfig(
            max_nodes=max_nodes,
            max_tokens=max_tokens,
            max_depth=1000,
            batch_size=1,
            enable_async=False,
            collect_metrics=True
        )

        result = _lib.initializeASTParser(ctypes.byref(self.config))
        if result != 0:
            raise RuntimeError(f"Failed to initialize AST parser: {result}")

    def parse(self, source_code: str) -> ASTParseResult:
        """
        Parse Python source code and return AST

        Args:
            source_code: Python source code as string

        Returns:
            ASTParseResult containing parsed nodes and metrics
        """
        source_bytes = source_code.encode('utf-8')
        source_length = len(source_bytes)

        nodes_ptr = ctypes.POINTER(ASTNode)()
        node_count = ctypes.c_uint32()
        metrics = ParserMetrics()

        result = _lib.parseSource(
            source_bytes,
            source_length,
            ctypes.byref(nodes_ptr),
            ctypes.byref(node_count),
            ctypes.byref(metrics)
        )

        if result != 0:
            return ASTParseResult(
                nodes=[],
                metrics={},
                success=False,
                error=f"CUDA error: {result}"
            )

        # Convert C array to Python list
        nodes_list = []
        for i in range(node_count.value):
            node = nodes_ptr[i]
            nodes_list.append({
                'type': node.node_type,
                'parent': node.parent_idx if node.parent_idx != 0xFFFFFFFF else None,
                'first_child': node.first_child if node.first_child != 0xFFFFFFFF else None,
                'next_sibling': node.next_sibling if node.next_sibling != 0xFFFFFFFF else None,
                'token_range': (node.token_start, node.token_end),
                'location': (node.line_number, node.col_number),
                'confidence': node.confidence,
            })

        metrics_dict = {
            'tokenization_time_ms': metrics.tokenization_time_ms,
            'parsing_time_ms': metrics.parsing_time_ms,
            'total_time_ms': metrics.total_time_ms,
            'nodes_created': metrics.nodes_created,
            'tokens_processed': metrics.tokens_processed,
            'gpu_utilization': metrics.gpu_utilization,
        }

        return ASTParseResult(
            nodes=nodes_list,
            metrics=metrics_dict,
            success=True
        )

    def __del__(self):
        _lib.cleanupASTParser()


class CUDAEmbeddingGenerator:
    """GPU-accelerated code embedding generator"""

    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 768,
        max_sequence_length: int = 512,
        pretrained_embeddings: Optional[np.ndarray] = None
    ):
        self.config = EmbeddingConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            batch_size=32,
            dropout_rate=0.1,
            use_fp16=False
        )

        # Prepare embeddings pointer
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocab_size, embedding_dim)
            embeddings_ptr = pretrained_embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            embeddings_ptr = None

        result = _lib.initializeEmbeddingGenerator(
            ctypes.byref(self.config),
            embeddings_ptr
        )

        if result != 0:
            raise RuntimeError(f"Failed to initialize embedding generator: {result}")

    def encode(
        self,
        token_sequences: List[List[int]],
        return_metrics: bool = True
    ) -> EmbeddingResult:
        """
        Generate embeddings for token sequences

        Args:
            token_sequences: List of token ID sequences
            return_metrics: Whether to return performance metrics

        Returns:
            EmbeddingResult containing embeddings and metrics
        """
        batch_size = len(token_sequences)
        max_len = self.config.max_sequence_length

        # Prepare input arrays
        token_ids = np.zeros((batch_size, max_len), dtype=np.uint32)
        sequence_lengths = np.zeros(batch_size, dtype=np.uint32)

        for i, seq in enumerate(token_sequences):
            seq_len = min(len(seq), max_len)
            token_ids[i, :seq_len] = seq[:seq_len]
            sequence_lengths[i] = seq_len

        # Prepare output
        embeddings_out = np.zeros(
            (batch_size, self.config.embedding_dim),
            dtype=np.float32
        )

        metrics = EmbeddingMetrics()

        result = _lib.generateEmbeddings(
            token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            sequence_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            batch_size,
            embeddings_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(metrics) if return_metrics else None
        )

        if result != 0:
            return EmbeddingResult(
                embeddings=np.array([]),
                metrics={},
                success=False,
                error=f"CUDA error: {result}"
            )

        metrics_dict = {}
        if return_metrics:
            metrics_dict = {
                'encoding_time_ms': metrics.encoding_time_ms,
                'total_time_ms': metrics.total_time_ms,
                'sequences_processed': metrics.sequences_processed,
                'throughput_seq_per_sec': metrics.throughput_seq_per_sec,
                'gpu_memory_used_mb': metrics.gpu_memory_used_mb,
            }

        return EmbeddingResult(
            embeddings=embeddings_out,
            metrics=metrics_dict,
            success=True
        )

    def find_similar(
        self,
        query_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find top-K most similar candidates for each query

        Args:
            query_embeddings: Query embeddings [num_queries x embedding_dim]
            candidate_embeddings: Candidate embeddings [num_candidates x embedding_dim]
            top_k: Number of top results to return

        Returns:
            Tuple of (indices, scores) arrays
        """
        num_queries = query_embeddings.shape[0]
        num_candidates = candidate_embeddings.shape[0]

        results = (SimilarityResult * (num_queries * top_k))()
        metrics = EmbeddingMetrics()

        result = _lib.computeSimilarity(
            query_embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_queries,
            candidate_embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_candidates,
            results,
            top_k,
            ctypes.byref(metrics)
        )

        if result != 0:
            raise RuntimeError(f"Similarity computation failed: {result}")

        # Convert results to numpy arrays
        indices = np.zeros((num_queries, top_k), dtype=np.uint32)
        scores = np.zeros((num_queries, top_k), dtype=np.float32)

        for i in range(num_queries):
            for j in range(top_k):
                idx = i * top_k + j
                indices[i, j] = results[idx].match_idx
                scores[i, j] = results[idx].similarity_score

        return indices, scores

    def __del__(self):
        _lib.cleanupEmbeddingGenerator()


# Utility functions

def is_cuda_available() -> bool:
    """Check if CUDA acceleration is available"""
    return not _use_cpu_fallback


def get_cuda_device_info() -> Dict[str, any]:
    """Get CUDA device information"""
    if _use_cpu_fallback:
        return {
            'available': False,
            'device_name': 'CPU Fallback',
            'compute_capability': None,
            'total_memory_mb': 0,
        }

    # This would query actual CUDA device properties
    return {
        'available': True,
        'device_name': 'NVIDIA GPU',
        'compute_capability': '7.0+',
        'total_memory_mb': 0,  # Would be queried from CUDA
    }


def benchmark_performance(
    parser: CUDAASTParser,
    source_samples: List[str],
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark parser performance

    Args:
        parser: CUDAASTParser instance
        source_samples: List of source code samples
        num_runs: Number of benchmark runs

    Returns:
        Performance statistics
    """
    import time

    times = []

    for _ in range(num_runs):
        start = time.perf_counter()

        for source in source_samples:
            result = parser.parse(source)
            if not result.success:
                raise RuntimeError(f"Parse failed: {result.error}")

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'throughput_files_per_sec': len(source_samples) / (np.mean(times) / 1000),
    }


# Example usage
if __name__ == "__main__":
    # Test AST parser
    parser = CUDAASTParser()

    test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, x, y):
        return x + y
"""

    result = parser.parse(test_code)

    if result.success:
        print(f"Parsed {len(result.nodes)} nodes")
        print(f"Metrics: {result.metrics}")
    else:
        print(f"Parse failed: {result.error}")

    # Test embedding generator
    embedder = CUDAEmbeddingGenerator(
        vocab_size=1000,
        embedding_dim=128
    )

    token_sequences = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
    ]

    emb_result = embedder.encode(token_sequences)

    if emb_result.success:
        print(f"Generated embeddings: {emb_result.embeddings.shape}")
        print(f"Metrics: {emb_result.metrics}")
    else:
        print(f"Embedding failed: {emb_result.error}")
