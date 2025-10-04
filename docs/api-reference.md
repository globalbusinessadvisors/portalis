# API Reference

REST API reference for Portalis translation service.

## Base URL

```
Production: https://api.portalis.dev
Staging: https://staging-api.portalis.dev
Local: http://localhost:8080
```

## Authentication

All API requests require authentication via API key:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.portalis.dev/api/v1/translate
```

### Get API Key

```bash
# Register account
curl -X POST https://api.portalis.dev/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure123"}'

# Response includes API key
{
  "api_key": "pk_live_abc123xyz",
  "user_id": "user_123"
}
```

## Endpoints

### POST /api/v1/translate

Translate Python code to Rust/WASM.

**Request**:
```json
{
  "python_code": "def add(a: int, b: int) -> int:\n    return a + b",
  "mode": "nemo",
  "temperature": 0.2,
  "include_metrics": true,
  "format": "wasm"
}
```

**Response**:
```json
{
  "request_id": "req_abc123",
  "rust_code": "pub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}",
  "wasm_bytes": "<base64-encoded>",
  "confidence": 0.98,
  "metrics": {
    "total_time_ms": 145.2,
    "gpu_utilization": 0.85,
    "tokens_processed": 42
  },
  "status": "success"
}
```

**Parameters**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `python_code` | string | Yes | - | Python source code |
| `mode` | string | No | `"pattern"` | Translation mode: `"pattern"` or `"nemo"` |
| `temperature` | float | No | `0.2` | Sampling temperature (0.0-1.0) |
| `include_metrics` | boolean | No | `false` | Include performance metrics |
| `format` | string | No | `"rust"` | Output format: `"rust"`, `"wasm"`, or `"both"` |

**Error Response**:
```json
{
  "error": {
    "code": "UNSUPPORTED_FEATURE",
    "message": "Metaclasses not supported",
    "details": {
      "line": 42,
      "feature": "metaclass"
    }
  }
}
```

### POST /api/v1/batch

Batch translation of multiple files.

**Request**:
```json
{
  "files": [
    {
      "name": "module1.py",
      "content": "def func1(): pass"
    },
    {
      "name": "module2.py",
      "content": "def func2(): pass"
    }
  ],
  "mode": "nemo"
}
```

**Response**:
```json
{
  "batch_id": "batch_xyz789",
  "results": [
    {
      "name": "module1.py",
      "rust_code": "...",
      "status": "success"
    },
    {
      "name": "module2.py",
      "rust_code": "...",
      "status": "success"
    }
  ],
  "summary": {
    "total": 2,
    "succeeded": 2,
    "failed": 0
  }
}
```

### GET /api/v1/status/:request_id

Check translation status.

**Response**:
```json
{
  "request_id": "req_abc123",
  "status": "completed",
  "progress": 100,
  "created_at": "2025-10-03T12:00:00Z",
  "completed_at": "2025-10-03T12:00:01Z"
}
```

### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "nemo": "available",
    "cuda": "available"
  }
}
```

## Rate Limiting

**Limits**:
- Free tier: 100 requests/hour
- Pro tier: 1,000 requests/hour
- Enterprise: Custom limits

**Headers**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1638360000
```

**Error Response** (429):
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 3600 seconds."
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request |
| `UNAUTHORIZED` | 401 | Missing/invalid API key |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `UNSUPPORTED_FEATURE` | 422 | Python feature not supported |
| `TRANSLATION_FAILED` | 500 | Translation error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## SDKs

### Python SDK

```bash
pip install portalis-sdk
```

```python
from portalis import Client

client = Client(api_key="pk_live_abc123")

# Translate code
result = client.translate(
    python_code="def add(a, b): return a + b",
    mode="nemo"
)

print(result.rust_code)
```

### JavaScript SDK

```bash
npm install @portalis/sdk
```

```javascript
const { PortalisClient } = require('@portalis/sdk');

const client = new PortalisClient({
  apiKey: 'pk_live_abc123'
});

const result = await client.translate({
  pythonCode: 'def add(a, b): return a + b',
  mode: 'nemo'
});

console.log(result.rustCode);
```

## Webhooks

Configure webhooks for async operations:

**Setup**:
```bash
curl -X POST https://api.portalis.dev/api/v1/webhooks \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"url": "https://yourapp.com/webhook", "events": ["translation.completed"]}'
```

**Payload**:
```json
{
  "event": "translation.completed",
  "request_id": "req_abc123",
  "data": {
    "rust_code": "...",
    "status": "success"
  }
}
```

## See Also

- [Getting Started](getting-started.md)
- [CLI Reference](cli-reference.md)
- [Python SDK Documentation](https://docs.portalis.dev/sdk/python)
