"""
NIM (NVIDIA Inference Microservices) Optimizations

Implements advanced optimizations for NIM deployment:
- Connection pooling for reduced latency
- Response compression for bandwidth efficiency
- Request batching and queuing
- Caching strategies
- Load balancing
"""

import asyncio
import gzip
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from threading import Lock
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class NIMOptimizationConfig:
    """Configuration for NIM optimizations."""

    # Connection pooling
    max_connections: int = 100
    min_connections: int = 10
    connection_timeout_seconds: int = 30
    connection_ttl_seconds: int = 300  # 5 minutes

    # Caching
    enable_cache: bool = True
    cache_size_mb: int = 1024  # 1GB cache
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_eviction_policy: str = "lru"  # "lru", "lfu", "fifo"

    # Compression
    enable_compression: bool = True
    compression_level: int = 6  # 1-9, higher = better compression but slower
    compression_min_size_bytes: int = 1024  # Only compress if > 1KB

    # Batching
    enable_batching: bool = True
    max_batch_size: int = 32
    batch_timeout_ms: int = 50  # Wait up to 50ms to form batch

    # Load balancing
    enable_load_balancing: bool = True
    load_balancing_strategy: str = "least_loaded"  # "round_robin", "least_loaded", "response_time"

    # Retry and circuit breaker
    max_retries: int = 3
    retry_backoff_ms: int = 100
    circuit_breaker_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_timeout_seconds: int = 60


class ConnectionPool:
    """
    Connection pool for NIM services.

    Maintains a pool of reusable connections to reduce overhead.
    """

    def __init__(self, config: NIMOptimizationConfig):
        self.config = config
        self.connections: List[Any] = []
        self.available: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.stats = {
            'total_created': 0,
            'total_reused': 0,
            'total_closed': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }

    async def initialize(self):
        """Pre-create minimum number of connections."""
        for _ in range(self.config.min_connections):
            conn = await self._create_connection()
            self.connections.append(conn)
            await self.available.put(conn)

        logger.info(f"Connection pool initialized with {self.config.min_connections} connections")

    async def _create_connection(self) -> Any:
        """Create a new connection."""
        # In production, create actual HTTP/gRPC connection
        connection = {
            'id': self.stats['total_created'],
            'created_at': time.time(),
            'last_used': time.time(),
            'use_count': 0
        }
        self.stats['total_created'] += 1
        return connection

    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        try:
            # Try to get available connection
            conn = await asyncio.wait_for(
                self.available.get(),
                timeout=self.config.connection_timeout_seconds
            )

            # Check if connection is still valid
            if time.time() - conn['created_at'] > self.config.connection_ttl_seconds:
                # Connection expired, create new one
                conn = await self._create_connection()
                self.stats['pool_misses'] += 1
            else:
                self.stats['pool_hits'] += 1

            conn['last_used'] = time.time()
            conn['use_count'] += 1
            self.stats['total_reused'] += 1

            return conn

        except asyncio.TimeoutError:
            # Pool exhausted, create new connection
            async with self.lock:
                if len(self.connections) < self.config.max_connections:
                    conn = await self._create_connection()
                    self.connections.append(conn)
                    self.stats['pool_misses'] += 1
                    return conn
                else:
                    raise RuntimeError("Connection pool exhausted")

    async def release(self, conn: Any):
        """Release connection back to pool."""
        await self.available.put(conn)

    async def close(self):
        """Close all connections."""
        async with self.lock:
            for conn in self.connections:
                # Close actual connection here
                self.stats['total_closed'] += 1
            self.connections.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            **self.stats,
            'pool_size': len(self.connections),
            'available': self.available.qsize(),
            'hit_rate': self.stats['pool_hits'] / max(1, self.stats['pool_hits'] + self.stats['pool_misses'])
        }


class ResponseCache:
    """
    LRU cache for NIM responses.

    Caches translation results to avoid redundant computation.
    """

    def __init__(self, config: NIMOptimizationConfig):
        self.config = config
        self.cache: OrderedDict = OrderedDict()
        self.lock = Lock()
        self.max_size_bytes = config.cache_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0
        }

    def _get_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key from request."""
        # Hash the request to create deterministic key
        key_data = json.dumps(request, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        key = self._get_cache_key(request)

        with self.lock:
            if key in self.cache:
                entry = self.cache[key]

                # Check if entry is expired
                if time.time() - entry['timestamp'] > self.config.cache_ttl_seconds:
                    # Expired, remove from cache
                    del self.cache[key]
                    self.current_size_bytes -= entry['size']
                    self.stats['misses'] += 1
                    return None

                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats['hits'] += 1
                return entry['response']
            else:
                self.stats['misses'] += 1
                return None

    def put(self, request: Dict[str, Any], response: Dict[str, Any]):
        """Put response in cache."""
        key = self._get_cache_key(request)
        response_size = len(json.dumps(response).encode())

        with self.lock:
            # Evict entries if necessary
            while self.current_size_bytes + response_size > self.max_size_bytes and self.cache:
                # Remove least recently used
                _, evicted = self.cache.popitem(last=False)
                self.current_size_bytes -= evicted['size']
                self.stats['evictions'] += 1

            # Add to cache
            self.cache[key] = {
                'response': response,
                'timestamp': time.time(),
                'size': response_size
            }
            self.current_size_bytes += response_size
            self.stats['size_bytes'] = self.current_size_bytes

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(1, total_requests)

        return {
            **self.stats,
            'entries': len(self.cache),
            'hit_rate': hit_rate,
            'size_mb': self.current_size_bytes / (1024 * 1024)
        }

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0


class ResponseCompressor:
    """
    Response compression for bandwidth optimization.

    Compresses large responses using gzip.
    """

    def __init__(self, config: NIMOptimizationConfig):
        self.config = config
        self.stats = {
            'compressed_count': 0,
            'uncompressed_count': 0,
            'bytes_saved': 0,
            'compression_ratio': 0.0
        }

    def compress(self, response: Dict[str, Any]) -> Tuple[bytes, bool]:
        """
        Compress response if beneficial.

        Returns:
            Tuple of (compressed_data, was_compressed)
        """
        if not self.config.enable_compression:
            data = json.dumps(response).encode()
            return data, False

        # Serialize response
        data = json.dumps(response).encode()

        # Only compress if above threshold
        if len(data) < self.config.compression_min_size_bytes:
            self.stats['uncompressed_count'] += 1
            return data, False

        # Compress
        compressed = gzip.compress(data, compresslevel=self.config.compression_level)

        # Update statistics
        bytes_saved = len(data) - len(compressed)
        self.stats['compressed_count'] += 1
        self.stats['bytes_saved'] += bytes_saved
        self.stats['compression_ratio'] = (
            self.stats['compression_ratio'] * (self.stats['compressed_count'] - 1) +
            len(compressed) / len(data)
        ) / self.stats['compressed_count']

        return compressed, True

    def decompress(self, data: bytes, is_compressed: bool) -> Dict[str, Any]:
        """Decompress response if needed."""
        if is_compressed:
            data = gzip.decompress(data)

        return json.loads(data.decode())

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        total = self.stats['compressed_count'] + self.stats['uncompressed_count']
        return {
            **self.stats,
            'compression_rate': self.stats['compressed_count'] / max(1, total),
            'avg_compression_ratio': self.stats['compression_ratio']
        }


class RequestBatcher:
    """
    Batch multiple requests for efficient processing.

    Collects requests and submits them as batches to NIM.
    """

    def __init__(self, config: NIMOptimizationConfig):
        self.config = config
        self.pending_requests: List[Tuple[Dict[str, Any], asyncio.Future]] = []
        self.lock = asyncio.Lock()
        self.batch_event = asyncio.Event()
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0
        }

    async def submit(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit request for batched processing.

        Args:
            request: Translation request

        Returns:
            Translation response
        """
        future = asyncio.Future()

        async with self.lock:
            self.pending_requests.append((request, future))
            self.stats['total_requests'] += 1

            # Trigger batch if full
            if len(self.pending_requests) >= self.config.max_batch_size:
                self.batch_event.set()

        # Wait for batch processing
        try:
            result = await asyncio.wait_for(
                future,
                timeout=self.config.batch_timeout_ms / 1000.0
            )
            return result
        except asyncio.TimeoutError:
            # Timeout, process batch anyway
            self.batch_event.set()
            return await future

    async def process_batches(self, processor_fn):
        """
        Background task to process batches.

        Args:
            processor_fn: Async function to process batch
        """
        while True:
            try:
                # Wait for batch or timeout
                await asyncio.wait_for(
                    self.batch_event.wait(),
                    timeout=self.config.batch_timeout_ms / 1000.0
                )
            except asyncio.TimeoutError:
                pass

            # Process pending requests
            async with self.lock:
                if not self.pending_requests:
                    self.batch_event.clear()
                    continue

                # Extract batch
                batch = self.pending_requests[:self.config.max_batch_size]
                self.pending_requests = self.pending_requests[self.config.max_batch_size:]

                if not self.pending_requests:
                    self.batch_event.clear()

            # Process batch
            requests = [req for req, _ in batch]
            futures = [fut for _, fut in batch]

            try:
                results = await processor_fn(requests)

                # Resolve futures
                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)

                # Update stats
                self.stats['total_batches'] += 1
                self.stats['avg_batch_size'] = (
                    (self.stats['avg_batch_size'] * (self.stats['total_batches'] - 1) + len(batch))
                    / self.stats['total_batches']
                )

            except Exception as e:
                # Reject all futures
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return {
            **self.stats,
            'pending': len(self.pending_requests)
        }


class NIMOptimizer:
    """
    Complete NIM optimization suite.

    Combines all optimization techniques for maximum performance.
    """

    def __init__(self, config: Optional[NIMOptimizationConfig] = None):
        self.config = config or NIMOptimizationConfig()

        # Initialize components
        self.connection_pool = ConnectionPool(self.config)
        self.cache = ResponseCache(self.config)
        self.compressor = ResponseCompressor(self.config)
        self.batcher = RequestBatcher(self.config)

        self.initialized = False

    async def initialize(self):
        """Initialize all optimization components."""
        await self.connection_pool.initialize()
        self.initialized = True
        logger.info("NIM optimizer initialized")

    async def translate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute optimized translation request.

        Args:
            request: Translation request

        Returns:
            Translation response
        """
        if not self.initialized:
            await self.initialize()

        # Check cache first
        cached_response = self.cache.get(request)
        if cached_response:
            logger.debug("Cache hit")
            return cached_response

        # Submit to batcher
        if self.config.enable_batching:
            response = await self.batcher.submit(request)
        else:
            # Direct execution
            conn = await self.connection_pool.acquire()
            try:
                response = await self._execute_request(conn, request)
            finally:
                await self.connection_pool.release(conn)

        # Cache response
        if self.config.enable_cache:
            self.cache.put(request, response)

        return response

    async def _execute_request(self, conn: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single request."""
        # Mock implementation - replace with actual NIM API call
        await asyncio.sleep(0.01)  # Simulate network delay
        return {
            'rust_code': 'fn mock() -> i32 { 0 }',
            'confidence': 0.95,
            'metadata': {'connection_id': conn['id']}
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'connection_pool': self.connection_pool.get_stats(),
            'cache': self.cache.get_stats(),
            'compression': self.compressor.get_stats(),
            'batching': self.batcher.get_stats(),
            'config': {
                'max_connections': self.config.max_connections,
                'cache_size_mb': self.config.cache_size_mb,
                'max_batch_size': self.config.max_batch_size,
                'compression_enabled': self.config.enable_compression
            }
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    async def main():
        config = NIMOptimizationConfig(
            max_connections=50,
            cache_size_mb=512,
            enable_compression=True,
            max_batch_size=16
        )

        optimizer = NIMOptimizer(config)
        await optimizer.initialize()

        # Example request
        request = {
            'python_code': 'def hello(): print("Hello")',
            'options': {}
        }

        response = await optimizer.translate(request)
        print(f"Response: {response}")

        # Get performance report
        report = optimizer.get_performance_report()
        print(f"\nPerformance Report:")
        print(json.dumps(report, indent=2))

    asyncio.run(main())
