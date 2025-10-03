# Portalis NIM Microservice Dockerfile
# Based on NVIDIA NGC PyTorch container with NeMo support
#
# Multi-stage build for optimized image size
# Target: <500MB compressed

# Stage 1: Builder
FROM nvcr.io/nvidia/pytorch:24.01-py3 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (for WASM compilation)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup target add wasm32-unknown-unknown

# Set working directory
WORKDIR /build

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install application
RUN pip install --no-cache-dir -e .

# Generate gRPC code
RUN python -m grpc_tools.protoc \
    -I./grpc \
    --python_out=./grpc \
    --grpc_python_out=./grpc \
    ./grpc/translation.proto || true

# Stage 2: Runtime
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Metadata
LABEL maintainer="Portalis Team"
LABEL description="NVIDIA NIM microservice for Python-to-Rust translation"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash portalis && \
    mkdir -p /app /models /data && \
    chown -R portalis:portalis /app /models /data

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cargo/bin/rustc /usr/local/bin/
COPY --from=builder /root/.cargo/bin/cargo /usr/local/bin/

# Copy application code
COPY --chown=portalis:portalis . .

# Copy gRPC generated files
COPY --from=builder /build/grpc/*_pb2*.py ./grpc/ || true

# Switch to non-root user
USER portalis

# Expose ports
# 8000: REST API
# 50051: gRPC
# 9090: Prometheus metrics
EXPOSE 8000 50051 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
