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

# Copy and install requirements with BuildKit cache
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip==25.2 && \
    pip install -r requirements.txt && \
    pip list

# ======= STAGE 2: Application =======
FROM deps as app

# Copy application code
COPY env/ ./env/
COPY *.py ./
COPY entrypoint.sh ./
# Create directories (checkpoints and results will be volumes)
RUN mkdir -p ./checkpoints ./results ./tensorboard ./models

# Make entrypoint executable
RUN chmod +x entrypoint.sh

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

# Create cache directory for Numba
RUN mkdir -p /tmp/numba_cache

# ======= STAGE 3: Production =======
FROM app as production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import numpy; print('OK')" || exit 1

# Use entrypoint to precompute features if needed
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - can be overridden in docker-compose
CMD ["python", "-u", "train.py", "--timesteps", "1000000", "--n_envs", "1"]