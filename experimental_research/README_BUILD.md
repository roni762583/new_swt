# EfficientZero Research Container - Build & Development Guide

**Status**: ✅ Production-Ready Docker Container with Build Cache Optimization  
**Location**: `/Users/shmuelzbaida/Desktop/Aharon2025/new_muzero/SWT/experimental_research`  
**Build System**: Professional Makefile + Docker Compose + Multi-stage Dockerfile  
**Standards**: Strict CLAUDE.md compliance with production-grade architecture

## 🚀 Quick Start

### Build & Run (Single Command)
```bash
cd /Users/shmuelzbaida/Desktop/Aharon2025/new_muzero/SWT/experimental_research

# Quick start: setup + build + validate + train
make quick-start

# Quick test: validate system + run component tests
make quick-test
```

### Step-by-Step Development
```bash
# 1. Setup development environment
make setup

# 2. Build Docker image with cache optimization
make build

# 3. Validate all systems
make validate

# 4. Run component tests  
make test

# 5. Start training experiment
make train
```

## 🏗️ Build Cache Optimization

### Multi-Stage Docker Build
Leverages your existing SWT container build cache:

**Stage 1 - Builder**:
- Reuses system dependencies from parent SWT container
- BuildKit cache mounts for Python packages
- Optimized layer ordering for maximum cache hits

**Stage 2 - Runtime**:
- Minimal production image
- Pre-compiled packages from builder stage
- Security hardened with non-root user

### Cache Strategy
```bash
# Build with cache (recommended for development)
make build

# Build without cache (for fresh releases)
make build-no-cache

# View cache layers
docker history efficientzero-research:latest
```

## 📊 Architecture Components

### Core EfficientZero Implementation

**✅ Value-Prefix Network** (`value_prefix_network.py`):
- **Multi-architecture support**: Transformer, TCN, LSTM, 1D-CNN
- **Recommendation**: Transformer for forex patterns (global attention)
- **Production features**: Automatic shape validation, error handling, metrics tracking

**✅ Self-Supervised Consistency Loss** (`consistency_loss.py`):
- **SimSiam-style implementation** with stop gradient
- **WST-aware loss** for market feature consistency  
- **Adaptive temperature scheduling** for optimal training
- **Production metrics**: Similarity tracking, uniformity measures

**✅ Model-Based Off-Policy Correction** (`off_policy_correction.py`):
- **Adaptive horizons** based on market volatility and data age
- **Forex optimizations**: Market regime detection, volatility adjustment
- **AMDDP1 compatibility** with 1% drawdown penalty system
- **Production robustness**: Fallback mechanisms, comprehensive error handling

### Professional Development Environment

**✅ Docker Container**:
- Multi-stage build optimization
- Build cache leveraging from parent SWT container
- Security hardened with non-root user
- Health checks and logging configuration

**✅ Docker Compose**:
- Primary trainer service with resource limits
- Monitoring service for real-time dashboard
- Volume management for persistent data
- Network configuration for service communication

**✅ Production Makefile**:
- 30+ professional commands for all development tasks
- Build cache optimization
- Testing and validation automation
- Container orchestration and monitoring

## 🔧 Development Workflow

### Daily Development Commands
```bash
# Start development session
make up                    # Start all services
make logs-trainer         # Monitor training logs  
make dashboard            # Open monitoring dashboard

# Development tasks
make shell                # Access container shell
make test                 # Run component tests
make lint                 # Code quality checks

# Container management
make restart              # Restart services
make status               # View service status
make down                 # Stop all services
```

### Experiment Management
```bash
# Run custom experiment
make experiment
# Interactive prompts for:
# - Experiment name
# - Episode count
# - Architecture choice (transformer/tcn/lstm/conv1d)

# Run training locally (no Docker)
make train-local

# Backup results
make backup
```

### Monitoring & Debugging
```bash
# Real-time monitoring
make logs                 # All service logs
make logs-trainer         # Trainer-specific logs
make logs-monitor         # Monitor service logs

# Debug mode
make debug               # Interactive debug container
make shell-root          # Root shell access

# Performance monitoring
make dashboard           # Opens http://localhost:8080
# - Training metrics: http://localhost:5003
# - TensorBoard: http://localhost:6008
```

## 📈 Build Performance Optimizations

### Cache Hit Optimization
1. **Layer Ordering**: Dependencies copied before source code
2. **BuildKit Cache Mounts**: Persistent pip cache across builds
3. **Multi-stage**: Builder artifacts cached separately from runtime
4. **Parent Cache Reuse**: Leverages existing SWT container layers

### Expected Build Performance
- **First Build**: ~15-20 minutes (full dependency compilation)
- **Cached Build**: ~2-3 minutes (90%+ cache hits)
- **Code-only Changes**: ~30 seconds (source layer only)

### Build Statistics
```bash
# View image layers and sizes
docker history efficientzero-research:latest

# Build with detailed output
DOCKER_BUILDKIT=1 docker build --progress=plain -t efficientzero-research:latest .

# Analyze build cache
docker system df
```

## 🧪 Testing & Validation

### Comprehensive Validation Suite
```bash
# System validation (always run first)
make validate
# Validates:
# - Python environment and dependencies
# - SWT base system components  
# - EfficientZero enhancement components
# - System resources and hardware

# Component testing
make test
# Tests:
# - Value-prefix network architectures
# - Consistency loss implementations
# - Off-policy correction algorithms
```

### Quality Assurance
```bash
# Code quality checks
make lint                 # Flake8 + MyPy
make test                 # Component tests
make validate            # System validation

# Full quality pipeline
make quick-test          # Validation + Testing
```

## 🔍 Professional Code Standards

### CLAUDE.md Compliance
- **No monkey patches**: Complete implementations only
- **Production-ready**: Type hints, docstrings, error handling
- **Fail fast**: Explicit error handling, no silent fallbacks
- **DRY principle**: No code duplication
- **Input validation**: Comprehensive parameter checking

### Code Quality Features
- **Type annotations**: All functions fully typed
- **Comprehensive docstrings**: Clear specifications
- **Error handling**: Explicit failure modes
- **Logging**: Production-grade logging throughout
- **Resource management**: Proper cleanup and memory management

### Testing Standards
- **Unit tests**: For all core components
- **Integration tests**: System-wide validation
- **Performance tests**: Resource usage validation
- **Error path testing**: Failure mode validation

## 📦 Container Specifications

### Resource Requirements
```yaml
# Development (docker-compose.yml)
resources:
  limits:
    memory: 10G      # Sufficient for Transformer + WST
    cpus: '6.0'      # Reserve cores for intensive research
  reservations:
    memory: 6G       # Minimum guaranteed memory
    cpus: '4.0'      # Minimum guaranteed cores
```

### Port Configuration
- **5003**: Training dashboard
- **6008**: TensorBoard
- **8080**: Monitoring dashboard

### Volume Mounts
```yaml
volumes:
  - ./checkpoints:/app/experimental_research/checkpoints    # Model persistence
  - ./logs:/app/experimental_research/logs                  # Training logs
  - ./results:/app/experimental_research/results            # Experiment results
  - ./data:/app/experimental_research/data                  # Training data
```

## 🚀 Production Deployment

### Production Build
```bash
# Production-optimized build
make build

# Multi-platform build (if needed)
docker buildx build --platform linux/amd64,linux/arm64 -t efficientzero-research:latest .

# Push to registry (configure DOCKER_REGISTRY)
make push
```

### Environment Configuration
```bash
# Set production environment variables
export EFFICIENTZERO_ENABLED=1
export TRAINING_MODE=production
export MONITORING_ENABLED=1
export ARCHITECTURE_MODE=transformer

# Run production training
docker-compose -f docker-compose.prod.yml up -d
```

### Monitoring & Maintenance
```bash
# Production monitoring
make logs-trainer         # Training progress
make dashboard            # Performance metrics
make status               # Service health

# Maintenance tasks
make backup               # Backup results/checkpoints
make clean                # Clean Docker resources
make info                 # System information
```

## 🔧 Customization & Extension

### Configuration Management
- **Main config**: `configs/efficientzero_experiment_config.json`
- **Docker environment**: `docker-compose.yml`
- **Build arguments**: `Dockerfile` build args

### Adding New Components
1. **Create module**: Follow existing patterns in `value_prefix_network.py`
2. **Add tests**: Include test functions for validation
3. **Update main**: Add to `efficientzero_main.py` validation
4. **Update Makefile**: Add relevant build/test commands

### Custom Architectures
```python
# Add new architecture to value_prefix_network.py
SUPPORTED_ARCHITECTURES = ['transformer', 'tcn', 'lstm', 'conv1d', 'your_new_arch']

def _build_your_new_arch(self, **kwargs):
    # Implementation here
```

## 📚 References & Standards

### Build System References
- **Docker Best Practices**: Multi-stage builds, layer optimization
- **BuildKit Documentation**: Cache mount optimization
- **Makefile Standards**: GNU Make best practices

### Code Quality Standards
- **CLAUDE.md**: Professional development requirements
- **PEP 8**: Python code style
- **Type Hints**: PEP 484 type annotation standards

---

**Build System Status**: ✅ Production-Ready  
**Container Status**: ✅ Fully Optimized  
**Development Environment**: ✅ Professional-Grade  
**Ready for**: Training, Experimentation, Production Deployment