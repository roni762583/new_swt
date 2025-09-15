#!/bin/bash
# SWT Production Deployment Script
# Automated deployment with pre-checks, rollback, and monitoring
#
# Features:
# - Environment validation and pre-deployment checks
# - Automated Docker deployment with health verification
# - Configuration validation and security checks
# - Rollback mechanisms and failure recovery
# - Post-deployment monitoring and validation

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs/deployment"
DEPLOYMENT_LOG="$LOG_DIR/deploy_$(date +%Y%m%d_%H%M%S).log"

# Deployment settings
DEPLOYMENT_TIMEOUT=600  # 10 minutes
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10

# Create log directory
mkdir -p "$LOG_DIR"

# Logging functions
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$message"
    echo "$message" >> "$DEPLOYMENT_LOG"
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

log_step() {
    log "${PURPLE}[STEP]${NC} $1"
}

# Error handling
error_handler() {
    local line_no=$1
    log_error "Deployment failed at line $line_no"
    log_error "Check deployment log: $DEPLOYMENT_LOG"
    
    # Attempt rollback if deployment was in progress
    if [[ -f "$PROJECT_ROOT/.deployment_in_progress" ]]; then
        log_warn "Attempting automatic rollback..."
        rollback_deployment || log_error "Rollback failed - manual intervention required"
    fi
    
    exit 1
}

trap 'error_handler $LINENO' ERR

# Utility functions
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command not found: $1"
        exit 1
    fi
}

check_file() {
    if [[ ! -f "$1" ]]; then
        log_error "Required file not found: $1"
        exit 1
    fi
}

check_directory() {
    if [[ ! -d "$1" ]]; then
        log_error "Required directory not found: $1"
        exit 1
    fi
}

# Pre-deployment validation
validate_environment() {
    log_step "Validating deployment environment..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        check_command "$cmd"
    done
    
    # Check Docker daemon
    if ! docker info &>/dev/null; then
        log_error "Docker daemon not running"
        exit 1
    fi
    
    # Check available disk space (minimum 5GB)
    local available_space
    available_space=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    if [[ $available_space -lt 5242880 ]]; then  # 5GB in KB
        log_error "Insufficient disk space. Need at least 5GB free."
        exit 1
    fi
    
    # Check available memory (minimum 4GB)
    local available_memory
    available_memory=$(free -m | awk 'NR==2{print $7}')
    if [[ $available_memory -lt 4096 ]]; then
        log_error "Insufficient memory. Need at least 4GB available."
        exit 1
    fi
    
    log_success "Environment validation passed"
}

validate_configuration() {
    log_step "Validating configuration files..."
    
    # Check required configuration files
    check_file "$PROJECT_ROOT/config/live.yaml"
    check_file "$PROJECT_ROOT/docker/docker-compose.yml"
    check_file "$PROJECT_ROOT/docker/Dockerfile.live"
    
    # Validate environment variables
    local required_env_vars=(
        "SWT_OANDA_ACCOUNT_ID"
        "SWT_OANDA_API_TOKEN"
        "SWT_OANDA_ENVIRONMENT"
    )
    
    for var in "${required_env_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable not set: $var"
            log_info "Please set $var in your environment or .env file"
            exit 1
        fi
    done
    
    # Validate checkpoint file
    local checkpoint_path="${SWT_CHECKPOINT_PATH:-checkpoints/episode_13475.pth}"
    if [[ ! -f "$PROJECT_ROOT/$checkpoint_path" ]]; then
        log_error "Checkpoint file not found: $checkpoint_path"
        exit 1
    fi
    
    # Validate configuration syntax
    if ! python -c "
import yaml
import sys
try:
    with open('$PROJECT_ROOT/config/live.yaml') as f:
        yaml.safe_load(f)
    print('Configuration syntax valid')
except Exception as e:
    print(f'Configuration syntax error: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_error "Invalid configuration syntax in live.yaml"
        exit 1
    fi
    
    log_success "Configuration validation passed"
}

validate_security() {
    log_step "Running security validation..."
    
    # Check file permissions
    local sensitive_files=(
        "config/live.yaml"
        ".env"
    )
    
    for file in "${sensitive_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            local perms
            perms=$(stat -f "%A" "$PROJECT_ROOT/$file" 2>/dev/null || stat -c "%a" "$PROJECT_ROOT/$file" 2>/dev/null)
            if [[ "$perms" != "600" ]] && [[ "$perms" != "644" ]]; then
                log_warn "File $file has overly permissive permissions: $perms"
            fi
        fi
    done
    
    # Check for secrets in logs
    if grep -r -i "password\|secret\|key\|token" "$PROJECT_ROOT/logs" 2>/dev/null | grep -v "deployment.log"; then
        log_warn "Potential secrets found in log files"
    fi
    
    log_success "Security validation passed"
}

# Pre-deployment checks
run_pre_deployment_checks() {
    log_step "Running pre-deployment checks..."
    
    # Test OANDA API connectivity
    local oanda_url
    if [[ "${SWT_OANDA_ENVIRONMENT:-}" == "live" ]]; then
        oanda_url="https://api-fxtrade.oanda.com"
    else
        oanda_url="https://api-fxpractice.oanda.com"
    fi
    
    if ! curl -s --connect-timeout 10 --max-time 20 "$oanda_url/v3/accounts" \
         -H "Authorization: Bearer ${SWT_OANDA_API_TOKEN}" &>/dev/null; then
        log_error "OANDA API connectivity test failed"
        exit 1
    fi
    
    # Test checkpoint loading
    if ! python -c "
import torch
import sys
try:
    checkpoint = torch.load('$PROJECT_ROOT/${SWT_CHECKPOINT_PATH:-checkpoints/episode_13475.pth}', map_location='cpu')
    print('Checkpoint loaded successfully')
except Exception as e:
    print(f'Checkpoint loading failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_error "Checkpoint validation failed"
        exit 1
    fi
    
    log_success "Pre-deployment checks passed"
}

# Backup current deployment
backup_current_deployment() {
    log_step "Creating deployment backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup configuration
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        cp -r "$PROJECT_ROOT/config" "$backup_dir/"
    fi
    
    # Backup current containers (export if running)
    if docker ps | grep -q "swt-live-trader"; then
        log_info "Exporting current live trading container..."
        docker commit swt-live-trader "swt-live-trader:backup-$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Save current docker-compose state
    if [[ -f "$PROJECT_ROOT/docker/docker-compose.yml" ]]; then
        docker-compose -f "$PROJECT_ROOT/docker/docker-compose.yml" config > "$backup_dir/docker-compose-current.yml"
    fi
    
    echo "$backup_dir" > "$PROJECT_ROOT/.last_backup"
    log_success "Backup created: $backup_dir"
}

# Build Docker images
build_images() {
    log_step "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build live trading image
    log_info "Building live trading image..."
    docker build -f docker/Dockerfile.live -t swt-live-trader:latest \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg VERSION="${VERSION:-latest}" \
        . >> "$DEPLOYMENT_LOG" 2>&1
    
    # Tag with timestamp for rollback
    docker tag swt-live-trader:latest "swt-live-trader:deploy-$(date +%Y%m%d_%H%M%S)"
    
    log_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_step "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Mark deployment in progress
    touch .deployment_in_progress
    
    # Stop existing services gracefully
    if docker-compose -f docker/docker-compose.yml ps | grep -q "Up"; then
        log_info "Stopping existing services..."
        docker-compose -f docker/docker-compose.yml down --timeout 30
    fi
    
    # Deploy new services
    log_info "Starting new services..."
    docker-compose -f docker/docker-compose.yml up -d --remove-orphans
    
    log_success "Services deployed"
}

# Health checks
wait_for_health() {
    log_step "Waiting for services to become healthy..."
    
    local retries=0
    local max_retries=$HEALTH_CHECK_RETRIES
    
    while [[ $retries -lt $max_retries ]]; do
        log_info "Health check attempt $((retries + 1))/$max_retries"
        
        # Check live trader health
        if curl -s -f http://localhost:8080/health &>/dev/null; then
            local health_response
            health_response=$(curl -s http://localhost:8080/health | jq -r '.status' 2>/dev/null || echo "unknown")
            
            if [[ "$health_response" == "healthy" ]]; then
                log_success "Live trader service is healthy"
                break
            fi
        fi
        
        if [[ $retries -eq $((max_retries - 1)) ]]; then
            log_error "Health check failed after $max_retries attempts"
            return 1
        fi
        
        ((retries++))
        sleep $HEALTH_CHECK_INTERVAL
    done
    
    return 0
}

# Post-deployment validation
validate_deployment() {
    log_step "Validating deployment..."
    
    # Check container status
    local containers=("swt-live-trader" "swt-redis")
    for container in "${containers[@]}"; do
        if ! docker ps | grep -q "$container"; then
            log_error "Container not running: $container"
            return 1
        fi
    done
    
    # Check logs for errors
    log_info "Checking container logs for errors..."
    if docker logs swt-live-trader --tail 50 2>&1 | grep -i "error\|exception\|failed" | head -5; then
        log_warn "Errors found in container logs (showing first 5):"
    fi
    
    # Test API endpoints
    if ! curl -s -f http://localhost:8080/health | jq -r '.status' | grep -q "healthy"; then
        log_error "Health endpoint not responding correctly"
        return 1
    fi
    
    log_success "Deployment validation passed"
}

# Rollback function
rollback_deployment() {
    log_step "Rolling back deployment..."
    
    if [[ ! -f "$PROJECT_ROOT/.last_backup" ]]; then
        log_error "No backup found for rollback"
        return 1
    fi
    
    local backup_dir
    backup_dir=$(cat "$PROJECT_ROOT/.last_backup")
    
    if [[ ! -d "$backup_dir" ]]; then
        log_error "Backup directory not found: $backup_dir"
        return 1
    fi
    
    # Stop current services
    docker-compose -f docker/docker-compose.yml down --timeout 30 || true
    
    # Restore configuration
    if [[ -d "$backup_dir/config" ]]; then
        cp -r "$backup_dir/config" "$PROJECT_ROOT/"
    fi
    
    # Find and use backup image
    local backup_image
    backup_image=$(docker images | grep "swt-live-trader.*backup" | head -1 | awk '{print $1":"$2}')
    
    if [[ -n "$backup_image" ]]; then
        docker tag "$backup_image" swt-live-trader:latest
        log_info "Restored backup image: $backup_image"
    fi
    
    # Restart with restored configuration
    docker-compose -f docker/docker-compose.yml up -d
    
    log_success "Rollback completed"
}

# Cleanup function
cleanup_deployment() {
    log_step "Cleaning up deployment artifacts..."
    
    # Remove deployment marker
    rm -f "$PROJECT_ROOT/.deployment_in_progress"
    
    # Clean up old images (keep last 3)
    local old_images
    old_images=$(docker images | grep "swt-live-trader.*deploy" | tail -n +4 | awk '{print $3}')
    
    if [[ -n "$old_images" ]]; then
        echo "$old_images" | xargs docker rmi --force 2>/dev/null || true
        log_info "Cleaned up old deployment images"
    fi
    
    # Clean up old backups (keep last 5)
    find "$PROJECT_ROOT/backups" -maxdepth 1 -type d -name "20*" | sort | head -n -5 | xargs rm -rf 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Monitoring setup
setup_monitoring() {
    log_step "Setting up post-deployment monitoring..."
    
    # Create monitoring script
    cat > "$PROJECT_ROOT/scripts/monitor_deployment.sh" << 'EOF'
#!/bin/bash
# Post-deployment monitoring script

LOGFILE="/tmp/swt_monitoring.log"

while true; do
    timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    
    # Check container health
    if ! docker ps | grep -q "swt-live-trader.*Up"; then
        echo "[$timestamp] ALERT: Live trader container not running" >> "$LOGFILE"
    fi
    
    # Check API health
    if ! curl -s -f http://localhost:8080/health &>/dev/null; then
        echo "[$timestamp] ALERT: Health endpoint not responding" >> "$LOGFILE"
    fi
    
    # Check memory usage
    memory_usage=$(docker stats --no-stream --format "table {{.MemPerc}}" swt-live-trader 2>/dev/null | tail -1 | sed 's/%//')
    if [[ "$memory_usage" =~ ^[0-9]+$ ]] && [[ $memory_usage -gt 80 ]]; then
        echo "[$timestamp] ALERT: High memory usage: ${memory_usage}%" >> "$LOGFILE"
    fi
    
    sleep 300  # Check every 5 minutes
done
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/monitor_deployment.sh"
    
    # Start monitoring in background
    nohup "$PROJECT_ROOT/scripts/monitor_deployment.sh" &
    echo $! > "$PROJECT_ROOT/.monitoring_pid"
    
    log_success "Monitoring started (PID: $(cat "$PROJECT_ROOT/.monitoring_pid"))"
}

# Main deployment function
main() {
    log_info "=== SWT Production Deployment Starting ==="
    log_info "Deployment log: $DEPLOYMENT_LOG"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Load environment variables if .env exists
    if [[ -f .env ]]; then
        set -a
        source .env
        set +a
        log_info "Loaded environment variables from .env"
    fi
    
    # Pre-deployment phase
    validate_environment
    validate_configuration
    validate_security
    run_pre_deployment_checks
    
    # Deployment phase
    backup_current_deployment
    build_images
    deploy_services
    
    # Validation phase
    if ! wait_for_health; then
        log_error "Health check failed - initiating rollback"
        rollback_deployment
        exit 1
    fi
    
    if ! validate_deployment; then
        log_error "Deployment validation failed - initiating rollback"
        rollback_deployment
        exit 1
    fi
    
    # Post-deployment phase
    cleanup_deployment
    setup_monitoring
    
    log_success "=== SWT Production Deployment Completed Successfully ==="
    log_info "Services running:"
    docker-compose -f docker/docker-compose.yml ps
    
    log_info "Health check URL: http://localhost:8080/health"
    log_info "Monitoring log: /tmp/swt_monitoring.log"
    log_info "Deployment log: $DEPLOYMENT_LOG"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "health")
        curl -s http://localhost:8080/health | jq .
        ;;
    "logs")
        docker-compose -f docker/docker-compose.yml logs -f "${2:-swt-live-trader}"
        ;;
    "status")
        docker-compose -f docker/docker-compose.yml ps
        ;;
    "stop")
        docker-compose -f docker/docker-compose.yml down
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health|logs|status|stop}"
        exit 1
        ;;
esac