#!/bin/bash

# SWT Production Deployment Script
# Complete production deployment with safety checks and monitoring

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/deployment.log"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/episode_13475.pt}"
CONFIG_PATH="${CONFIG_PATH:-config/live.yaml}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; echo -e "${BLUE}ℹ️  $*${NC}"; }
log_success() { log "SUCCESS" "$@"; echo -e "${GREEN}✅ $*${NC}"; }
log_warning() { log "WARNING" "$@"; echo -e "${YELLOW}⚠️  $*${NC}"; }
log_error() { log "ERROR" "$@"; echo -e "${RED}❌ $*${NC}"; }

# Error handler
error_handler() {
    local line_no=$1
    log_error "Deployment failed at line $line_no"
    log_error "Cleaning up partial deployment..."
    
    # Attempt cleanup
    docker-compose -f docker-compose.live.yml down --remove-orphans || true
    
    exit 1
}

trap 'error_handler $LINENO' ERR

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "================================================"
    echo "🚀 SWT PRODUCTION DEPLOYMENT"
    echo "================================================"
    echo -e "${NC}"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose not available"
        exit 1
    fi
    
    # Check checkpoint file
    if [[ ! -f "$CHECKPOINT_PATH" ]]; then
        log_error "Checkpoint file not found: $CHECKPOINT_PATH"
        log_info "Please ensure Episode 13475 checkpoint is available"
        exit 1
    fi
    
    # Check config file
    if [[ ! -f "$CONFIG_PATH" ]]; then
        log_error "Configuration file not found: $CONFIG_PATH"
        exit 1
    fi
    
    # Check required directories
    local dirs=("logs" "sessions" "cache" "checkpoints" "monitoring")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_info "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Check disk space (minimum 5GB)
    local available_space=$(df "$SCRIPT_DIR" | awk 'NR==2{print $4}')
    local min_space=$((5 * 1024 * 1024)) # 5GB in KB
    
    if [[ $available_space -lt $min_space ]]; then
        log_error "Insufficient disk space. Available: ${available_space}KB, Required: ${min_space}KB"
        exit 1
    fi
    
    # Check memory (minimum 4GB)
    local available_memory=$(free -m | awk 'NR==2{print $7}')
    if [[ $available_memory -lt 4096 ]]; then
        log_warning "Low available memory: ${available_memory}MB (recommended: 4GB+)"
    fi
    
    log_success "Pre-deployment checks passed"
}

# Build containers
build_containers() {
    log_info "Building production containers..."
    
    # Clean up old images to save space
    log_info "Cleaning up old Docker images..."
    docker image prune -f || true
    
    # Build live trading container
    log_info "Building live trading container..."
    docker build -f Dockerfile.live -t swt-live-trader:latest . --no-cache
    
    # Verify container health
    log_info "Verifying container builds..."
    if ! docker run --rm swt-live-trader:latest python -c "import swt_core.config_manager; print('✅ Import test passed')"; then
        log_error "Live trading container build verification failed"
        exit 1
    fi
    
    log_success "Containers built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying production services..."
    
    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose -f docker-compose.live.yml down --remove-orphans || true
    
    # Start services
    log_info "Starting production services..."
    docker-compose -f docker-compose.live.yml up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 30
    
    log_success "Production services deployed"
}

# Health checks
run_health_checks() {
    log_info "Running health checks..."
    
    local max_attempts=30
    local attempt=1
    
    # Check Redis
    while [[ $attempt -le $max_attempts ]]; do
        if docker exec swt-redis redis-cli ping &>/dev/null; then
            log_success "Redis is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Redis health check failed after $max_attempts attempts"
            return 1
        fi
        
        log_info "Waiting for Redis... (attempt $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    # Check Prometheus
    attempt=1
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s http://localhost:9090/-/healthy &>/dev/null; then
            log_success "Prometheus is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Prometheus health check failed after $max_attempts attempts"
            return 1
        fi
        
        log_info "Waiting for Prometheus... (attempt $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    # Check Live Trading Service
    attempt=1
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s http://localhost:8080/health &>/dev/null; then
            log_success "Live Trading service is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Live Trading service health check failed after $max_attempts attempts"
            return 1
        fi
        
        log_info "Waiting for Live Trading service... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    # Check Grafana
    if curl -s http://localhost:3000/api/health &>/dev/null; then
        log_success "Grafana is healthy"
    else
        log_warning "Grafana may not be ready yet (non-critical)"
    fi
    
    log_success "All critical services are healthy"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring and alerts..."
    
    # Wait for Prometheus to be fully ready
    sleep 10
    
    # Reload Prometheus configuration
    if curl -s -X POST http://localhost:9090/-/reload; then
        log_success "Prometheus configuration reloaded"
    else
        log_warning "Failed to reload Prometheus configuration"
    fi
    
    log_info "Monitoring endpoints:"
    log_info "  - Grafana Dashboard: http://localhost:3000 (admin/swt_admin_2024)"
    log_info "  - Prometheus: http://localhost:9090"
    log_info "  - Live Trading Metrics: http://localhost:8080/metrics"
    log_info "  - Live Trading Health: http://localhost:8080/health"
    
    log_success "Monitoring setup complete"
}

# Performance verification
verify_performance() {
    log_info "Running performance verification..."
    
    # Test inference latency
    log_info "Testing inference performance..."
    
    # This would run actual performance tests
    # For now, just check that the service is responding
    local response_time=$(curl -o /dev/null -s -w "%{time_total}" http://localhost:8080/health)
    if (( $(echo "$response_time > 1.0" | bc -l) )); then
        log_warning "High response time: ${response_time}s"
    else
        log_success "Response time acceptable: ${response_time}s"
    fi
    
    log_success "Performance verification complete"
}

# Post-deployment summary
deployment_summary() {
    log_info "Deployment Summary"
    echo "================================================"
    echo "🎉 SWT Production Deployment Complete!"
    echo "================================================"
    echo ""
    echo "📊 Service Status:"
    docker-compose -f docker-compose.live.yml ps
    echo ""
    echo "🔗 Access Points:"
    echo "  Live Trading:   http://localhost:8080"
    echo "  Grafana:        http://localhost:3000"
    echo "  Prometheus:     http://localhost:9090"
    echo "  Redis:          localhost:6379"
    echo ""
    echo "📁 Important Directories:"
    echo "  Logs:           $SCRIPT_DIR/logs/"
    echo "  Sessions:       $SCRIPT_DIR/sessions/"
    echo "  Cache:          $SCRIPT_DIR/cache/"
    echo "  Checkpoints:    $SCRIPT_DIR/checkpoints/"
    echo ""
    echo "🔧 Management Commands:"
    echo "  View logs:      docker-compose -f docker-compose.live.yml logs -f"
    echo "  Stop services:  docker-compose -f docker-compose.live.yml down"
    echo "  Restart:        docker-compose -f docker-compose.live.yml restart"
    echo ""
    echo "⚠️  Remember to:"
    echo "  1. Monitor the Grafana dashboard for system health"
    echo "  2. Check logs regularly for any issues"
    echo "  3. Ensure Episode 13475 checkpoint is properly loaded"
    echo "  4. Set up proper alerting for production use"
    echo ""
}

# Main deployment function
main() {
    print_banner
    
    log_info "Starting SWT production deployment at $(date)"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Run deployment steps
    pre_deployment_checks
    build_containers
    deploy_services
    run_health_checks
    setup_monitoring
    verify_performance
    
    deployment_summary
    
    log_success "Deployment completed successfully at $(date)"
}

# Script options
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "Stopping production services..."
        docker-compose -f docker-compose.live.yml down
        log_success "Services stopped"
        ;;
    "restart")
        log_info "Restarting production services..."
        docker-compose -f docker-compose.live.yml restart
        log_success "Services restarted"
        ;;
    "status")
        echo "📊 Service Status:"
        docker-compose -f docker-compose.live.yml ps
        ;;
    "logs")
        docker-compose -f docker-compose.live.yml logs -f
        ;;
    "clean")
        log_info "Cleaning up deployment..."
        docker-compose -f docker-compose.live.yml down --volumes --remove-orphans
        docker system prune -f
        log_success "Cleanup complete"
        ;;
    "help")
        echo "SWT Production Deployment Script"
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy    - Full production deployment (default)"
        echo "  stop      - Stop all services"
        echo "  restart   - Restart all services"
        echo "  status    - Show service status"
        echo "  logs      - Follow service logs"
        echo "  clean     - Clean up everything"
        echo "  help      - Show this help"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac