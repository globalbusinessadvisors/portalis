"""
Locust Load Testing Scenarios

Realistic load testing for Portalis translation service:
- User behavior simulation
- Gradual ramp-up
- Stress testing
- Spike testing
"""

from locust import HttpUser, task, between, events
import random
import json
import time
from typing import Dict, Any


class PortalisUser(HttpUser):
    """
    Simulates a user of the Portalis translation service.

    User behaviors:
    - Translate small functions (70%)
    - Translate medium modules (25%)
    - Translate large files (5%)
    """

    wait_time = between(1, 5)  # Wait 1-5 seconds between requests

    def on_start(self):
        """Initialize user session."""
        self.client.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Portalis-LoadTest/1.0'
        }
        self.translation_count = 0
        self.start_time = time.time()

    @task(70)
    def translate_small_function(self):
        """
        Translate a small function (10-50 LOC).

        This is the most common operation.
        """
        code_size = random.randint(10, 50)
        python_code = self._generate_python_code(code_size)

        payload = {
            'python_code': python_code,
            'options': {
                'target': 'rust',
                'optimize': True
            }
        }

        with self.client.post(
            '/api/translate',
            json=payload,
            catch_response=True,
            name="translate_small"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get('confidence', 0) > 0.8:
                    response.success()
                    self.translation_count += 1
                else:
                    response.failure(f"Low confidence: {result.get('confidence')}")
            else:
                response.failure(f"Status {response.status_code}")

    @task(25)
    def translate_medium_module(self):
        """
        Translate a medium module (100-500 LOC).

        Represents typical module translation.
        """
        code_size = random.randint(100, 500)
        python_code = self._generate_python_code(code_size)

        payload = {
            'python_code': python_code,
            'options': {
                'target': 'rust',
                'optimize': True,
                'include_tests': True
            }
        }

        with self.client.post(
            '/api/translate',
            json=payload,
            catch_response=True,
            name="translate_medium"
        ) as response:
            if response.status_code == 200:
                self.translation_count += 1
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)
    def translate_large_file(self):
        """
        Translate a large file (1000+ LOC).

        Less common but important to test.
        """
        code_size = random.randint(1000, 2000)
        python_code = self._generate_python_code(code_size)

        payload = {
            'python_code': python_code,
            'options': {
                'target': 'rust',
                'optimize': True,
                'parallel': True
            }
        }

        with self.client.post(
            '/api/translate',
            json=payload,
            catch_response=True,
            name="translate_large"
        ) as response:
            if response.status_code == 200:
                self.translation_count += 1
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(10)
    def check_translation_status(self):
        """
        Check status of a previous translation.

        Simulates asynchronous workflow.
        """
        job_id = f"job-{random.randint(1, 1000)}"

        with self.client.get(
            f'/api/status/{job_id}',
            catch_response=True,
            name="check_status"
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def health_check(self):
        """Health check endpoint."""
        with self.client.get(
            '/health',
            catch_response=True,
            name="health_check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unhealthy: {response.status_code}")

    def _generate_python_code(self, num_lines: int) -> str:
        """Generate synthetic Python code for testing."""
        lines = []
        lines.append("# Auto-generated test code")
        lines.append("def generated_function(data):")
        lines.append("    '''Generated for load testing'''")

        for i in range(num_lines - 3):
            if i % 5 == 0:
                lines.append(f"    var_{i} = {random.randint(0, 100)}")
            elif i % 5 == 1:
                lines.append(f"    if var_{i-1} > 50:")
                lines.append(f"        result = var_{i-1} * 2")
            elif i % 5 == 2:
                lines.append(f"    else:")
                lines.append(f"        result = var_{i-2} + 10")
            else:
                lines.append(f"    # Processing step {i}")

        lines.append("    return result")

        return "\n".join(lines)

    def on_stop(self):
        """Cleanup on user stop."""
        elapsed = time.time() - self.start_time
        print(f"User completed {self.translation_count} translations in {elapsed:.1f}s")


class PowerUser(PortalisUser):
    """
    Power user with heavier usage patterns.

    - More large translations
    - Batch operations
    - Higher request rate
    """

    wait_time = between(0.5, 2)  # Faster requests

    @task(50)
    def batch_translate(self):
        """Batch translation of multiple files."""
        batch_size = random.randint(5, 20)
        files = []

        for i in range(batch_size):
            files.append({
                'filename': f'file_{i}.py',
                'code': self._generate_python_code(random.randint(50, 200))
            })

        payload = {
            'files': files,
            'options': {
                'target': 'rust',
                'parallel': True
            }
        }

        with self.client.post(
            '/api/batch-translate',
            json=payload,
            catch_response=True,
            name="batch_translate"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                self.translation_count += len(files)
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


# Event handlers for metrics collection

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("="*60)
    print("Portalis Load Test Starting")
    print("="*60)
    print(f"Target: {environment.host}")
    print(f"User classes: {[user.__name__ for user in environment.user_classes]}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops - print summary."""
    print("\n" + "="*60)
    print("Load Test Summary")
    print("="*60)

    stats = environment.stats
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    avg_response_time = stats.total.avg_response_time
    rps = stats.total.total_rps

    print(f"Total Requests:        {total_requests:,}")
    print(f"Total Failures:        {total_failures:,}")
    print(f"Failure Rate:          {total_failures/max(1,total_requests)*100:.2f}%")
    print(f"Avg Response Time:     {avg_response_time:.2f}ms")
    print(f"Requests per Second:   {rps:.2f}")

    # Check SLA compliance
    print("\nSLA Compliance:")
    p95 = stats.total.get_response_time_percentile(0.95)
    print(f"  P95 < 500ms:         {'✓ PASS' if p95 < 500 else '✗ FAIL'} ({p95:.0f}ms)")
    print(f"  Success Rate > 99%:  {'✓ PASS' if (1-total_failures/max(1,total_requests)) > 0.99 else '✗ FAIL'}")
    print(f"  QPS > 100:           {'✓ PASS' if rps > 100 else '✗ FAIL'} ({rps:.1f})")


# Custom shape classes for different load patterns

class StepLoadShape:
    """
    Step load pattern: gradually increase load in steps.

    Example: 10 users, then 50, then 100, then 200
    """

    def tick(self):
        run_time = self.get_run_time()

        if run_time < 60:
            return (10, 1)  # 10 users, spawn 1/sec
        elif run_time < 180:
            return (50, 2)  # 50 users, spawn 2/sec
        elif run_time < 300:
            return (100, 5)  # 100 users, spawn 5/sec
        elif run_time < 420:
            return (200, 10)  # 200 users, spawn 10/sec
        else:
            return None  # Stop test


class SpikeLoadShape:
    """
    Spike load pattern: sudden traffic spikes.

    Simulates traffic surges (e.g., from viral content).
    """

    def tick(self):
        run_time = self.get_run_time()

        if run_time < 60:
            return (50, 5)  # Baseline: 50 users
        elif run_time < 120:
            return (500, 50)  # SPIKE: 500 users
        elif run_time < 180:
            return (50, 5)  # Back to baseline
        elif run_time < 240:
            return (1000, 100)  # BIGGER SPIKE: 1000 users
        elif run_time < 300:
            return (50, 5)  # Back to baseline
        else:
            return None


class StressTestShape:
    """
    Stress test: push system to failure.

    Continuously increase load until system breaks.
    """

    def tick(self):
        run_time = self.get_run_time()

        # Increase users every 30 seconds
        users = int(run_time / 30) * 50
        spawn_rate = max(10, users // 10)

        if users > 2000:  # Cap at 2000 users
            return None

        return (users, spawn_rate)
