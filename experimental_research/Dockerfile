# EfficientZero-Enhanced SWT System - Production-Grade Multi-Stage Build
# Leverages build cache optimizations from parent SWT container
# Strict adherence to CLAUDE.md professional code standards

# Stage 1: Builder - Install all dependencies and compile libraries
FROM python:3.11-slim as builder

# Add labels to leverage SWT build cache
LABEL swt.cache.layer="efficientzero-system-deps"
LABEL swt.cache.preserve="true" 
LABEL swt.efficientzero.version="1.0"
LABEL description="EfficientZero experimental research - preserve for rebuild performance"

# Build arguments for reproducible builds
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILDKIT_INLINE_CACHE=1

# Install system dependencies (leverage cache from parent SWT container)
# CRITICAL: This layer should be preserved - contains heavy system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential build tools (same as SWT for cache reuse)
    gcc \
    g++ \
    make \
    cmake \
    # Python development headers
    python3-dev \
    # FFTW3 for wavelet transforms (required by Kymatio)
    libfftw3-dev \
    # BLAS/LAPACK for scientific computing
    libblas-dev \
    liblapack-dev \
    # OpenMP for parallel processing
    libomp-dev \
    # Git for source installs
    git \
    # Additional tools for EfficientZero research
    pkg-config \
    libhdf5-dev \
    # Clean up build cache
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    # Mark layer completion for cache identification
    && echo "EFFICIENTZERO_SYSTEM_DEPS_CACHE_LAYER_COMPLETE" > /tmp/efficientzero_cache_marker

# Set working directory
WORKDIR /build

# Copy requirements first for optimal Docker layer caching
COPY requirements.txt .

# Install Python dependencies with BuildKit cache mounts for maximum performance
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/tmp/pip-build-cache,sharing=locked \
    # Upgrade pip and build tools
    pip install --upgrade pip setuptools wheel && \
    # Install PyTorch first (largest dependency - reuse SWT cache layer!)
    pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
    # Install scientific/wavelet libraries (reuse SWT cache!)
    pip install pytorch-wavelets>=1.3.0 && \
    pip install kymatio>=0.3.0 && \
    pip install PyWavelets>=1.4.0 && \
    # Install gymnasium (reuse SWT cache)
    pip install gymnasium>=0.28.0 && \
    # Install EfficientZero-specific dependencies
    pip install einops>=0.6.0 && \
    pip install transformers>=4.30.0 && \
    pip install pytorch-lightning>=2.0.0 && \
    pip install pytorch-tcn>=1.0.0 && \
    # Install all remaining requirements
    pip install --no-cache-dir -r requirements.txt && \
    # Verify critical installations
    python -c "import torch, einops, transformers; print('✅ EfficientZero dependencies verified')"

# Stage 2: Production Runtime - Minimal runtime image
FROM python:3.11-slim as runtime

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Add production labels
LABEL swt.efficientzero.stage="production"
LABEL swt.efficientzero.build_date="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
LABEL maintainer="SWT Research Team"

# Install only runtime dependencies (minimal for production)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime libraries for scientific computing (same as SWT)
    libfftw3-double3 \
    libfftw3-single3 \
    libblas3 \
    liblapack3 \
    libgomp1 \
    # HDF5 runtime for data processing
    libhdf5-103 \
    # Process monitoring and debugging
    lsof \
    procps \
    htop \
    # Network tools for monitoring
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user for security (production best practice)
RUN useradd -m -u 1000 efficientzero-user && \
    mkdir -p /app/experimental_research && \
    chown -R efficientzero-user:efficientzero-user /app

# Copy Python packages from builder stage (pre-compiled and cached!)
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Set working directory
WORKDIR /app/experimental_research

# Copy the entire experimental research directory
COPY --chown=efficientzero-user:efficientzero-user . .

# Switch to non-root user (security best practice)
USER efficientzero-user

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/experimental_research

# Optimize for scientific computing (same as SWT for consistency)
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV FFTW_THREADS=4

# EfficientZero-specific optimizations
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV CUDA_LAUNCH_BLOCKING=0

# Expose ports for monitoring and dashboard
EXPOSE 5003 6008

# Health check - verify all critical systems work in production runtime
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD python -c "\
import torch; \
import einops; \
import transformers; \
import kymatio; \
import pytorch_wavelets; \
import gymnasium; \
print('🔬 EfficientZero System Health Check:'); \
print('✅ PyTorch:', torch.__version__); \
print('✅ Transformers:', transformers.__version__); \
print('✅ Kymatio WST: Ready'); \
print('✅ Gymnasium: Ready'); \
print('🚀 EfficientZero Research Container: ALL SYSTEMS OPERATIONAL')" \
    || exit 1

# Create necessary directories with proper permissions
RUN mkdir -p \
    checkpoints \
    logs \
    results \
    data \
    configs \
    monitoring

# Production startup script
COPY --chown=efficientzero-user:efficientzero-user docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Default command - run system validation and training
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "efficientzero_main.py", "--validate-system"]