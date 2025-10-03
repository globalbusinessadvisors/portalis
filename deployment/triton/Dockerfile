# Multi-stage Dockerfile for Portalis Triton Deployment
# Builds custom Triton image with NeMo and translation models

# Stage 1: Build Python dependencies
FROM python:3.10-slim as python-builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Build wheels
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# Stage 2: NeMo model preparation
FROM nvcr.io/nvidia/nemo:24.01.01 as nemo-builder

WORKDIR /nemo-models

# Copy model download scripts
COPY scripts/download-nemo-models.sh .

# Download and optimize NeMo models
RUN bash download-nemo-models.sh


# Stage 3: Final Triton image
FROM nvcr.io/nvidia/tritonserver:24.01-py3

LABEL maintainer="Portalis Team"
LABEL description="Triton Inference Server with Portalis Translation Models"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy Python wheels and install
COPY --from=python-builder /wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Install additional Python packages for translation
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    torch>=2.1.0 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0

# Copy NeMo models
COPY --from=nemo-builder /nemo-models /models/nemo/

# Create model repository structure
RUN mkdir -p /models/{translation_model,interactive_api,batch_processor}

# Copy Triton model configs and implementations
COPY models/translation_model/config.pbtxt /models/translation_model/
COPY models/translation_model/1/model.py /models/translation_model/1/
COPY models/interactive_api/config.pbtxt /models/interactive_api/
COPY models/interactive_api/1/model.py /models/interactive_api/1/
COPY models/batch_processor/config.pbtxt /models/batch_processor/

# Create warmup data
RUN mkdir -p /models/translation_model/1/warmup \
    /models/interactive_api/1/warmup

# Copy warmup samples
COPY warmup-data/simple_code.dat /models/translation_model/1/warmup/
COPY warmup-data/batch_code.dat /models/translation_model/1/warmup/
COPY warmup-data/interactive_sample.dat /models/interactive_api/1/warmup/
COPY warmup-data/target_lang.dat /models/interactive_api/1/warmup/

# Set up Python backend environment
RUN mkdir -p /opt/tritonserver/backends/python/envs/translation_env \
    /opt/tritonserver/backends/python/envs/interactive_env

# Create health check script
COPY scripts/health-check.sh /usr/local/bin/health-check
RUN chmod +x /usr/local/bin/health-check

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0,1
ENV TRITON_SERVER_CPU_ONLY=0
ENV OMP_NUM_THREADS=8
ENV NCCL_DEBUG=INFO
ENV MODEL_REPOSITORY=/models

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/health-check

# Set working directory
WORKDIR /opt/tritonserver

# Default command
CMD ["/opt/tritonserver/bin/tritonserver", \
     "--model-repository=/models", \
     "--strict-model-config=false", \
     "--log-verbose=1", \
     "--allow-gpu-metrics=true", \
     "--allow-cpu-metrics=true", \
     "--metrics-port=8002", \
     "--backend-directory=/opt/tritonserver/backends", \
     "--backend-config=python,shm-default-byte-size=16777216"]
