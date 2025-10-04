"""
Load Testing Script for Portalis Rust Transpiler

Tests concurrent translation requests, auto-scaling, and performance metrics.
Uses locust for load generation and testing.
"""

import json
import random
import time
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import logging

logger = logging.getLogger(__name__)


# Sample Python code snippets for testing
SAMPLE_CODE_SNIPPETS = [
    # Simple function
    """def add(a, b):
    return a + b""",

    # Fibonacci
    """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",

    # List comprehension
    """def squares(n):
    return [i**2 for i in range(n)]""",

    # Class definition
    """class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, x):
        self.result += x
        return self.result""",

    # Decorator
    """def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time: {end - start}")
        return result
    return wrapper""",

    # Generator
    """def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1""",

    # Context manager
    """class FileHandler:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = open(self.filename, 'r')
        return self.file

    def __exit__(self, *args):
        self.file.close()""",

    # Async function
    """async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()""",

    # Data processing
    """def process_data(data):
    filtered = [x for x in data if x > 0]
    sorted_data = sorted(filtered)
    return sum(sorted_data) / len(sorted_data)""",

    # Recursive function
    """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
]


class TranspilerUser(FastHttpUser):
    """User simulating transpiler API usage"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    @task(10)
    def translate_fast(self):
        """Fast translation (most common)"""
        code = random.choice(SAMPLE_CODE_SNIPPETS)

        with self.client.post(
            "/api/v1/translation/translate",
            json={
                "python_code": code,
                "mode": "fast",
                "temperature": 0.2,
                "include_alternatives": False
            },
            catch_response=True,
            name="translate_fast"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get("rust_code"):
                    response.success()
                else:
                    response.failure("No Rust code in response")
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)
    def translate_quality(self):
        """Quality translation"""
        code = random.choice(SAMPLE_CODE_SNIPPETS)

        with self.client.post(
            "/api/v1/translation/translate",
            json={
                "python_code": code,
                "mode": "quality",
                "temperature": 0.1,
                "include_alternatives": True
            },
            catch_response=True,
            name="translate_quality"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get("rust_code") and result.get("confidence", 0) > 0.8:
                    response.success()
                else:
                    response.failure("Low quality translation")
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    def translate_batch(self):
        """Batch translation"""
        # Select 3-5 random snippets
        snippets = random.sample(SAMPLE_CODE_SNIPPETS, k=random.randint(3, 5))

        with self.client.post(
            "/api/v1/translation/translate/batch",
            json={
                "source_files": snippets,
                "project_config": {"name": "test", "version": "1.0"},
                "optimization_level": "release",
                "compile_wasm": False
            },
            catch_response=True,
            name="translate_batch"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get("success_count", 0) > 0:
                    response.success()
                else:
                    response.failure("Batch translation failed")
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def health_check(self):
        """Health check endpoint"""
        with self.client.get("/health", name="health_check") as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def metrics(self):
        """Metrics endpoint"""
        self.client.get("/metrics", name="metrics")

    @task(1)
    def list_models(self):
        """List available models"""
        with self.client.get("/api/v1/translation/models", name="list_models") as response:
            if response.status_code == 200:
                result = response.json()
                if "models" in result:
                    response.success()
                else:
                    response.failure("No models in response")


class StressTestUser(FastHttpUser):
    """User for stress testing with higher load"""

    wait_time = between(0.1, 0.5)  # Minimal wait time for stress

    @task
    def rapid_fire_translate(self):
        """Rapid fire translations for stress testing"""
        code = random.choice(SAMPLE_CODE_SNIPPETS[:3])  # Use simpler snippets

        self.client.post(
            "/api/v1/translation/translate",
            json={
                "python_code": code,
                "mode": "fast"
            },
            name="stress_translate"
        )


# Custom event handlers for metrics collection

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log request metrics"""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test"""
    logger.info("Load test starting...")
    logger.info(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Finalize test and report results"""
    logger.info("Load test completed")

    stats = environment.stats
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")
    logger.info(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"P95 response time: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    logger.info(f"P99 response time: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    logger.info(f"Requests per second: {stats.total.total_rps:.2f}")


# Locust configuration via code
class LocustConfig:
    """Configuration for locust run"""

    # Standard load test
    NORMAL_USERS = 50
    NORMAL_SPAWN_RATE = 10
    NORMAL_DURATION = "10m"

    # Stress test
    STRESS_USERS = 200
    STRESS_SPAWN_RATE = 50
    STRESS_DURATION = "5m"

    # Spike test
    SPIKE_USERS = 500
    SPIKE_SPAWN_RATE = 100
    SPIKE_DURATION = "2m"


if __name__ == "__main__":
    print("Load Testing Configuration")
    print("=" * 50)
    print("\nRun with locust:")
    print("\n  Normal load test:")
    print(f"    locust -f load_test.py --users {LocustConfig.NORMAL_USERS} --spawn-rate {LocustConfig.NORMAL_SPAWN_RATE} --run-time {LocustConfig.NORMAL_DURATION} --host http://localhost:8000")
    print("\n  Stress test:")
    print(f"    locust -f load_test.py --users {LocustConfig.STRESS_USERS} --spawn-rate {LocustConfig.STRESS_SPAWN_RATE} --run-time {LocustConfig.STRESS_DURATION} --host http://localhost:8000 StressTestUser")
    print("\n  Spike test:")
    print(f"    locust -f load_test.py --users {LocustConfig.SPIKE_USERS} --spawn-rate {LocustConfig.SPIKE_SPAWN_RATE} --run-time {LocustConfig.SPIKE_DURATION} --host http://localhost:8000")
    print("\nOr run with web UI:")
    print("    locust -f load_test.py --host http://localhost:8000")
    print("    Then open http://localhost:8089")
