// K6 Load Testing Script for Portalis Rust Transpiler
// Tests concurrent translation requests and validates auto-scaling

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const translationErrors = new Rate('translation_errors');
const translationDuration = new Trend('translation_duration');
const translationSuccess = new Counter('translation_success');
const translationFailure = new Counter('translation_failure');

// Test configuration
export const options = {
  stages: [
    // Ramp up to 10 users over 1 minute
    { duration: '1m', target: 10 },
    // Stay at 10 users for 2 minutes
    { duration: '2m', target: 10 },
    // Ramp up to 50 users over 2 minutes
    { duration: '2m', target: 50 },
    // Stay at 50 users for 3 minutes
    { duration: '3m', target: 50 },
    // Spike to 100 users for 1 minute
    { duration: '1m', target: 100 },
    // Ramp down to 0 over 1 minute
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000', 'p(99)<5000'], // 95% < 2s, 99% < 5s
    http_req_failed: ['rate<0.05'], // Error rate < 5%
    translation_errors: ['rate<0.05'],
    translation_duration: ['p(95)<1500'],
  },
};

// Sample Python code snippets
const sampleCode = [
  'def add(a, b):\n    return a + b',
  'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
  'def squares(n):\n    return [i**2 for i in range(n)]',
  'class Calculator:\n    def __init__(self):\n        self.result = 0\n    def add(self, x):\n        self.result += x',
  'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)',
];

// Base URL from environment or default
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  // Health check
  group('Health Check', function () {
    const res = http.get(`${BASE_URL}/health`);
    check(res, {
      'health check status is 200': (r) => r.status === 200,
      'health check response time < 500ms': (r) => r.timings.duration < 500,
    });
  });

  // Fast translation
  group('Fast Translation', function () {
    const payload = JSON.stringify({
      python_code: sampleCode[Math.floor(Math.random() * sampleCode.length)],
      mode: 'fast',
      temperature: 0.2,
      include_alternatives: false,
    });

    const params = {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: '30s',
    };

    const start = Date.now();
    const res = http.post(`${BASE_URL}/api/v1/translation/translate`, payload, params);
    const duration = Date.now() - start;

    translationDuration.add(duration);

    const success = check(res, {
      'translation status is 200': (r) => r.status === 200,
      'translation has rust_code': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.rust_code && body.rust_code.length > 0;
        } catch (e) {
          return false;
        }
      },
      'translation confidence > 0.8': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.confidence > 0.8;
        } catch (e) {
          return false;
        }
      },
    });

    if (success) {
      translationSuccess.add(1);
    } else {
      translationFailure.add(1);
      translationErrors.add(1);
    }
  });

  // Quality translation (less frequent)
  if (Math.random() < 0.3) {
    group('Quality Translation', function () {
      const payload = JSON.stringify({
        python_code: sampleCode[Math.floor(Math.random() * sampleCode.length)],
        mode: 'quality',
        temperature: 0.1,
        include_alternatives: true,
      });

      const params = {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: '60s',
      };

      const res = http.post(`${BASE_URL}/api/v1/translation/translate`, payload, params);

      check(res, {
        'quality translation status is 200': (r) => r.status === 200,
        'quality translation has high confidence': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.confidence > 0.9;
          } catch (e) {
            return false;
          }
        },
      });
    });
  }

  // Batch translation (rare)
  if (Math.random() < 0.1) {
    group('Batch Translation', function () {
      const payload = JSON.stringify({
        source_files: sampleCode.slice(0, 3),
        project_config: { name: 'k6-test', version: '1.0' },
        optimization_level: 'release',
        compile_wasm: false,
      });

      const params = {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: '120s',
      };

      const res = http.post(`${BASE_URL}/api/v1/translation/translate/batch`, payload, params);

      check(res, {
        'batch translation status is 200': (r) => r.status === 200,
        'batch has successful translations': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.success_count > 0;
          } catch (e) {
            return false;
          }
        },
      });
    });
  }

  // Metrics check
  if (Math.random() < 0.05) {
    group('Metrics', function () {
      const res = http.get(`${BASE_URL}/metrics`);
      check(res, {
        'metrics endpoint accessible': (r) => r.status === 200,
      });
    });
  }

  sleep(Math.random() * 2 + 1); // Random sleep 1-3 seconds
}

// Teardown function
export function teardown(data) {
  console.log('Load test completed');
  console.log(`Total successful translations: ${translationSuccess.count}`);
  console.log(`Total failed translations: ${translationFailure.count}`);
}
