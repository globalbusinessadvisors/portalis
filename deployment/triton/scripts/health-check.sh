#!/bin/bash
# Health check script for Triton container

set -e

# Check if Triton server is responding
if curl -f -s http://localhost:8000/v2/health/live > /dev/null 2>&1; then
    # Server is live, check if ready
    if curl -f -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
        echo "Triton server is healthy"
        exit 0
    else
        echo "Triton server is live but not ready"
        exit 1
    fi
else
    echo "Triton server is not responding"
    exit 1
fi
