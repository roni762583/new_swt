# UNIFIED PRODUCTION DOCKERFILE - PPO NEURAL NETWORK TRAINING
# Combines all optimizations: BuildKit, parallelization, JIT, GPU support

# Use BuildKit for efficient caching
# syntax=docker/dockerfile:1.4

FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ======= STAGE 1: Dependencies =======
FROM base as deps

# Copy and install requirements with caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Install additional optimizations
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    numba>=0.58.0 \
    ray>=2.7.0 \
    joblib>=1.3.0 \
    tqdm>=4.65.0

# ======= STAGE 2: Application =======
FROM deps as app

# Copy application code
COPY env/ ./env/
COPY *.py ./
COPY checkpoints/ ./checkpoints/
COPY results/ ./results/

# Set optimization environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # CPU optimization
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    VECLIB_MAXIMUM_THREADS=8 \
    NUMBA_NUM_THREADS=8 \
    # Numba settings
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_THREADING_LAYER=omp

# Create necessary directories
RUN mkdir -p /tmp/numba_cache checkpoints results

# ======= STAGE 3: Production =======
FROM app as production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import numpy; print('OK')" || exit 1

# Default command - can be overridden in docker-compose
CMD ["python", "-u", "train_optimized.py", "--episodes", "50", "--n_envs", "4"]