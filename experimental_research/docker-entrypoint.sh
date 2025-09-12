#!/bin/bash
# EfficientZero Research Container Entrypoint Script
# Professional-grade production entrypoint with comprehensive validation
# Strict adherence to CLAUDE.md code standards

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

readonly SCRIPT_NAME="$(basename "${0}")"
readonly SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
readonly APP_DIR="/app/experimental_research"
readonly LOG_DIR="${APP_DIR}/logs"
readonly CHECKPOINT_DIR="${APP_DIR}/checkpoints"
readonly RESULTS_DIR="${APP_DIR}/results"

# Logging functions
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} ${SCRIPT_NAME}: $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} ${SCRIPT_NAME}: $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} ${SCRIPT_NAME}: $*" >&2
}

log_debug() {
    if [[ "${DEBUG:-0}" == "1" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} ${SCRIPT_NAME}: $*" >&2
    fi
}

# =============================================================================
# SYSTEM VALIDATION FUNCTIONS  
# =============================================================================

validate_python_environment() {
    log_info "Validating Python environment..."
    
    # Check Python version
    local python_version
    python_version=$(python --version 2>&1)
    log_info "Python version: ${python_version}"
    
    # Validate critical imports
    python -c "
import sys
import torch
import numpy as np
import pandas as pd
import gymnasium
import kymatio
import pytorch_wavelets
import einops
import transformers

print('✅ Core libraries imported successfully')
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'Gymnasium: {gymnasium.__version__}')
print(f'Transformers: {transformers.__version__}')

# Test tensor operations
x = torch.randn(4, 4)
y = torch.matmul(x, x.transpose(0, 1))
print('✅ PyTorch tensor operations working')

# Test WST libraries
try:
    from kymatio.torch import Scattering1D
    scattering = Scattering1D(J=2, shape=(256,))
    print('✅ Kymatio WST library operational')
except Exception as e:
    print(f'⚠️  Kymatio warning: {e}')

# Test EfficientZero components
try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    encoder_layer = TransformerEncoderLayer(d_model=64, nhead=8)
    transformer = TransformerEncoder(encoder_layer, num_layers=2)
    print('✅ Transformer architecture ready')
except Exception as e:
    raise RuntimeError(f'Transformer validation failed: {e}')
"
    
    log_info "✅ Python environment validation complete"
}

validate_file_permissions() {
    log_info "Validating file permissions and directory structure..."
    
    # Check required directories
    local required_dirs=(
        "${LOG_DIR}"
        "${CHECKPOINT_DIR}" 
        "${RESULTS_DIR}"
        "${APP_DIR}/configs"
        "${APP_DIR}/monitoring"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "${dir}" ]]; then
            log_info "Creating directory: ${dir}"
            mkdir -p "${dir}"
        fi
        
        if [[ ! -w "${dir}" ]]; then
            log_error "Directory not writable: ${dir}"
            exit 1
        fi
    done
    
    log_info "✅ Directory structure validation complete"
}

validate_system_resources() {
    log_info "Validating system resources..."
    
    # Check memory
    local memory_mb
    memory_mb=$(free -m | awk '/^Mem:/ {print $2}')
    log_info "Available memory: ${memory_mb}MB"
    
    if (( memory_mb < 2048 )); then
        log_warn "Low memory detected: ${memory_mb}MB (recommended: 4GB+)"
    fi
    
    # Check CPU cores
    local cpu_cores
    cpu_cores=$(nproc)
    log_info "Available CPU cores: ${cpu_cores}"
    
    if (( cpu_cores < 2 )); then
        log_warn "Low CPU cores detected: ${cpu_cores} (recommended: 4+)"
    fi
    
    # Check disk space
    local disk_space
    disk_space=$(df -h "${APP_DIR}" | awk 'NR==2 {print $4}')
    log_info "Available disk space: ${disk_space}"
    
    log_info "✅ System resource validation complete"
}

validate_efficientzero_components() {
    log_info "Validating EfficientZero components..."
    
    # Check if EfficientZero modules can be imported
    python -c "
import sys
sys.path.insert(0, '${APP_DIR}')

try:
    from efficientzero_trainer import EfficientZeroSWTTrainer
    from value_prefix_network import SWTValuePrefixNetwork
    from consistency_loss import SWTConsistencyLoss
    print('✅ EfficientZero components imported successfully')
except ImportError as e:
    print(f'⚠️  EfficientZero components not yet implemented: {e}')
    print('This is expected during initial development phase')

# Test SWT base components
try:
    from swt_models.swt_stochastic_networks import create_swt_stochastic_muzero_network
    from swt_core.swt_mcts import create_swt_stochastic_mcts
    print('✅ Base SWT components accessible')
except ImportError as e:
    raise RuntimeError(f'Base SWT components failed: {e}')
"
    
    log_info "✅ EfficientZero component validation complete"
}

# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

initialize_logging() {
    log_info "Initializing logging system..."
    
    # Create log files with timestamps
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    
    export LOG_FILE="${LOG_DIR}/efficientzero_${timestamp}.log"
    export ERROR_LOG="${LOG_DIR}/efficientzero_error_${timestamp}.log"
    
    # Create log files
    touch "${LOG_FILE}" "${ERROR_LOG}"
    
    log_info "Log files initialized:"
    log_info "  Main log: ${LOG_FILE}"
    log_info "  Error log: ${ERROR_LOG}"
}

initialize_environment() {
    log_info "Initializing EfficientZero environment..."
    
    # Set default environment variables if not provided
    export EFFICIENTZERO_ENABLED="${EFFICIENTZERO_ENABLED:-1}"
    export TRAINING_MODE="${TRAINING_MODE:-research}"
    export MONITORING_ENABLED="${MONITORING_ENABLED:-1}"
    export ARCHITECTURE_MODE="${ARCHITECTURE_MODE:-transformer}"
    
    # Performance optimization
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
    
    # EfficientZero-specific settings
    export CONSISTENCY_LOSS_ENABLED="${CONSISTENCY_LOSS_ENABLED:-true}"
    export VALUE_PREFIX_ENABLED="${VALUE_PREFIX_ENABLED:-true}"
    export OFF_POLICY_CORRECTION_ENABLED="${OFF_POLICY_CORRECTION_ENABLED:-true}"
    
    log_info "Environment configuration:"
    log_info "  Training Mode: ${TRAINING_MODE}"
    log_info "  Architecture: ${ARCHITECTURE_MODE}"
    log_info "  Consistency Loss: ${CONSISTENCY_LOSS_ENABLED}"
    log_info "  Value Prefix: ${VALUE_PREFIX_ENABLED}"
    log_info "  Off-Policy Correction: ${OFF_POLICY_CORRECTION_ENABLED}"
}

# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

run_system_validation() {
    log_info "🔬 Starting EfficientZero Research System Validation..."
    
    validate_python_environment
    validate_file_permissions
    validate_system_resources
    validate_efficientzero_components
    
    log_info "🚀 System validation complete - EfficientZero Research ready!"
}

execute_main_command() {
    local cmd=("$@")
    
    log_info "Executing main command: ${cmd[*]}"
    
    # Change to app directory
    cd "${APP_DIR}"
    
    # Execute with logging
    exec "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
}

# =============================================================================
# SIGNAL HANDLING
# =============================================================================

cleanup() {
    log_info "Received termination signal, cleaning up..."
    
    # Kill child processes
    pkill -P $$ || true
    
    # Final log message
    log_info "EfficientZero Research Container shutdown complete"
    
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    log_info "🔬 EfficientZero Research Container Starting..."
    log_info "Container: $(hostname)"
    log_info "User: $(whoami)"
    log_info "Working Directory: $(pwd)"
    log_info "Arguments: $*"
    
    # Initialize systems
    initialize_logging
    initialize_environment
    
    # Always run system validation
    run_system_validation
    
    # Execute main command or default
    if [[ $# -eq 0 ]]; then
        log_info "No command provided, running default system test"
        execute_main_command python efficientzero_system_test.py
    else
        execute_main_command "$@"
    fi
}

# Execute main function with all arguments
main "$@"