#!/bin/bash
# SWT Container Entrypoint Script
# Handles container initialization, health checks, and command routing

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

# Environment validation
validate_environment() {
    log_info "Validating environment configuration..."
    
    local errors=0
    
    # Check Python environment
    if ! python --version &>/dev/null; then
        log_error "Python not found"
        ((errors++))
    else
        log_info "Python version: $(python --version)"
    fi
    
    # Check required directories
    for dir in data logs config; do
        if [[ ! -d "/app/$dir" ]]; then
            log_warn "Creating missing directory: /app/$dir"
            mkdir -p "/app/$dir"
        fi
    done
    
    # Check environment-specific requirements
    if [[ "${SWT_ENVIRONMENT:-}" == "production" ]]; then
        # Production environment checks
        required_vars=(
            "SWT_OANDA_ACCOUNT_ID"
            "SWT_OANDA_API_TOKEN" 
            "SWT_CHECKPOINT_PATH"
        )
        
        for var in "${required_vars[@]}"; do
            if [[ -z "${!var:-}" ]]; then
                log_error "Required environment variable not set: $var"
                ((errors++))
            fi
        done
        
        # Check checkpoint file exists
        if [[ -n "${SWT_CHECKPOINT_PATH:-}" ]] && [[ ! -f "${SWT_CHECKPOINT_PATH}" ]]; then
            log_error "Checkpoint file not found: ${SWT_CHECKPOINT_PATH}"
            ((errors++))
        fi
        
    elif [[ "${SWT_ENVIRONMENT:-}" == "training" ]]; then
        # Training environment checks
        if command -v nvidia-smi &>/dev/null; then
            log_info "GPU info: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        else
            log_warn "NVIDIA GPU not detected (CPU-only training)"
        fi
        
        # Check for training data
        if [[ ! -d "/app/data" ]] || [[ -z "$(ls -A /app/data 2>/dev/null)" ]]; then
            log_warn "Training data directory is empty"
        fi
    fi
    
    if [[ $errors -gt 0 ]]; then
        log_error "Environment validation failed with $errors errors"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# System health check
health_check() {
    log_info "Running system health check..."
    
    # Memory check
    local memory_usage
    memory_usage=$(python -c "import psutil; print(psutil.virtual_memory().percent)")
    log_info "Memory usage: ${memory_usage}%"
    
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        log_warn "High memory usage: ${memory_usage}%"
    fi
    
    # Disk space check
    local disk_usage
    disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    log_info "Disk usage: ${disk_usage}%"
    
    if [[ $disk_usage -gt 85 ]]; then
        log_warn "High disk usage: ${disk_usage}%"
    fi
    
    # Network connectivity check (for live trading)
    if [[ "${SWT_ENVIRONMENT:-}" == "production" ]]; then
        log_info "Checking OANDA API connectivity..."
        local oanda_url
        if [[ "${SWT_OANDA_ENVIRONMENT:-}" == "live" ]]; then
            oanda_url="https://api-fxtrade.oanda.com"
        else
            oanda_url="https://api-fxpractice.oanda.com"
        fi
        
        if curl -s --connect-timeout 5 --max-time 10 "$oanda_url/health" >/dev/null; then
            log_success "OANDA API connectivity OK"
        else
            log_warn "OANDA API connectivity check failed"
        fi
    fi
}

# Process management setup
setup_process_limits() {
    log_info "Setting up process limits..."
    
    # Set memory limits if specified
    if [[ -n "${SWT_MAX_MEMORY:-}" ]]; then
        # Convert memory limit (e.g., "2G" to bytes)
        local memory_bytes
        memory_bytes=$(echo "${SWT_MAX_MEMORY}" | sed 's/G/000000000/;s/M/000000/')
        
        # Set virtual memory limit
        ulimit -v "$((memory_bytes / 1024))"  # ulimit expects KB
        log_info "Set memory limit: ${SWT_MAX_MEMORY}"
    fi
    
    # Set CPU limits via cgroups (if available)
    if [[ -n "${SWT_MAX_CPU:-}" ]] && [[ -w "/sys/fs/cgroup/cpu/cpu.cfs_quota_us" ]]; then
        local cpu_quota=$((${SWT_MAX_CPU} * 100000))  # 100000 = 1 CPU
        echo "$cpu_quota" > /sys/fs/cgroup/cpu/cpu.cfs_quota_us
        log_info "Set CPU limit: ${SWT_MAX_CPU} cores"
    fi
    
    # Set process limits
    if [[ -n "${SWT_MAX_PROCESSES:-}" ]]; then
        ulimit -u "${SWT_MAX_PROCESSES}"
        log_info "Set process limit: ${SWT_MAX_PROCESSES}"
    fi
    
    # Set file descriptor limits
    if [[ -n "${SWT_MAX_FILES:-}" ]]; then
        ulimit -n "${SWT_MAX_FILES}"
        log_info "Set file descriptor limit: ${SWT_MAX_FILES}"
    fi
}

# Signal handling for graceful shutdown
setup_signal_handlers() {
    log_info "Setting up signal handlers..."
    
    # Function to handle shutdown signals
    shutdown_handler() {
        log_info "Received shutdown signal, initiating graceful shutdown..."
        
        # Send SIGTERM to all child processes
        pkill -TERM -P $$
        
        # Wait for processes to terminate
        local timeout=30
        local count=0
        
        while pgrep -P $$ > /dev/null && [[ $count -lt $timeout ]]; do
            sleep 1
            ((count++))
        done
        
        if pgrep -P $$ > /dev/null; then
            log_warn "Forcing termination of remaining processes"
            pkill -KILL -P $$
        fi
        
        log_success "Shutdown complete"
        exit 0
    }
    
    # Register signal handlers
    trap shutdown_handler SIGTERM SIGINT SIGQUIT
}

# Start health check server
start_health_server() {
    log_info "Starting health check server on port 8080..."
    
    python -c "
import http.server
import socketserver
import json
import psutil
from datetime import datetime

class HealthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(),
                'disk_percent': psutil.disk_usage('/app').percent,
                'environment': '${SWT_ENVIRONMENT:-unknown}'
            }
            
            self.wfile.write(json.dumps(health_data).encode())
        else:
            self.send_response(404)
            self.end_headers()

try:
    with socketserver.TCPServer(('', 8080), HealthHandler) as httpd:
        httpd.serve_forever()
except Exception as e:
    print(f'Health server failed: {e}')
" &
    
    local health_pid=$!
    log_info "Health check server started (PID: $health_pid)"
}

# Main application commands
run_live_trading() {
    log_info "Starting SWT Live Trading System..."
    
    # Validate live trading requirements
    if [[ -z "${SWT_OANDA_ACCOUNT_ID:-}" ]] || [[ -z "${SWT_OANDA_API_TOKEN:-}" ]]; then
        log_error "OANDA credentials not configured"
        exit 1
    fi
    
    if [[ ! -f "${SWT_CHECKPOINT_PATH:-}" ]]; then
        log_error "Checkpoint file not found: ${SWT_CHECKPOINT_PATH:-}"
        exit 1
    fi
    
    # Start health server
    start_health_server
    
    # Start live trading
    exec python -m swt_live.main \
        --config "${SWT_CONFIG_FILE:-config/live.yaml}" \
        --checkpoint "${SWT_CHECKPOINT_PATH}" \
        --log-level "${SWT_LOG_LEVEL:-INFO}"
}

run_training() {
    log_info "Starting SWT Training System..."
    
    # Check for training data
    if [[ ! -d "/app/data" ]] || [[ -z "$(ls -A /app/data 2>/dev/null)" ]]; then
        log_error "Training data not found in /app/data"
        exit 1
    fi
    
    # Start health server
    start_health_server
    
    # Start training
    exec python -m swt_training.main \
        --config "${SWT_CONFIG_FILE:-config/training.yaml}" \
        --data-dir "/app/data" \
        --checkpoint-dir "/app/checkpoints" \
        --log-level "${SWT_LOG_LEVEL:-INFO}" \
        --num-episodes "${SWT_NUM_EPISODES:-500000}"
}

run_validation() {
    log_info "Running system validation..."
    
    # Run comprehensive system tests
    python -m pytest test_live_trading_system.py -v --tb=short
    
    log_success "Validation complete"
}

run_benchmark() {
    log_info "Running performance benchmarks..."
    
    # Run performance tests
    python -m scripts.benchmark_system \
        --config "${SWT_CONFIG_FILE:-config/live.yaml}" \
        --duration 300  # 5 minutes
    
    log_success "Benchmark complete"
}

# Main execution logic
main() {
    log_info "=== SWT Container Starting ==="
    log_info "Environment: ${SWT_ENVIRONMENT:-unknown}"
    log_info "Command: ${1:-default}"
    
    # Setup
    validate_environment
    setup_process_limits
    setup_signal_handlers
    health_check
    
    # Route to appropriate command
    case "${1:-live-trade}" in
        "live-trade"|"live")
            run_live_trading
            ;;
        "train"|"training")
            run_training
            ;;
        "validate"|"test")
            run_validation
            ;;
        "benchmark"|"bench")
            run_benchmark
            ;;
        "health")
            log_info "Container health check passed"
            ;;
        "bash"|"shell")
            log_info "Starting interactive shell..."
            exec /bin/bash
            ;;
        *)
            log_error "Unknown command: ${1:-}"
            log_info "Available commands: live-trade, train, validate, benchmark, health, bash"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"