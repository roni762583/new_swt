# EfficientZero Experimental Research - Production Build & Development
# Professional Makefile with build cache optimization and quality standards
# Strict adherence to CLAUDE.md professional code standards

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Configuration
PROJECT_NAME := efficientzero-research
VERSION := 1.0.0
BUILD_DATE := $(shell date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Docker configuration
DOCKER_REGISTRY := 
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_TAG := $(VERSION)
DOCKERFILE := Dockerfile

# Build cache optimization
BUILDKIT_INLINE_CACHE := 1
DOCKER_BUILDKIT := 1

# Development configuration
DEV_COMPOSE_FILE := docker-compose.yml
PROD_COMPOSE_FILE := docker-compose.prod.yml

# Directories
LOGS_DIR := logs
RESULTS_DIR := results
CHECKPOINTS_DIR := checkpoints
DATA_DIR := data
CONFIGS_DIR := configs

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Export Docker buildkit
export DOCKER_BUILDKIT

##@ Help
.PHONY: help
help: ## Display this help message
	@echo -e "$(BLUE)EfficientZero Experimental Research$(NC)"
	@echo -e "$(BLUE)Production-grade build and development system$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development
.PHONY: setup
setup: ## Setup development environment
	@echo -e "$(BLUE)🔧 Setting up EfficientZero development environment...$(NC)"
	@mkdir -p $(LOGS_DIR) $(RESULTS_DIR) $(CHECKPOINTS_DIR) $(DATA_DIR) $(CONFIGS_DIR)
	@echo -e "$(GREEN)✅ Development directories created$(NC)"
	@if command -v python3 >/dev/null 2>&1; then \
		echo -e "$(BLUE)📦 Installing Python dependencies...$(NC)"; \
		python3 -m pip install -r requirements.txt; \
		echo -e "$(GREEN)✅ Python dependencies installed$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠️  Python3 not found, skipping dependency installation$(NC)"; \
	fi

.PHONY: validate
validate: ## Run system validation
	@echo -e "$(BLUE)🔬 Running EfficientZero system validation...$(NC)"
	@python3 efficientzero_main.py --validate-system
	@echo -e "$(GREEN)✅ System validation complete$(NC)"

.PHONY: test
test: ## Run component tests
	@echo -e "$(BLUE)🧪 Running component tests...$(NC)"
	@python3 -m pytest value_prefix_network.py::test_value_prefix_network -v || python3 value_prefix_network.py
	@python3 -m pytest consistency_loss.py::test_consistency_loss -v || python3 consistency_loss.py
	@python3 -m pytest off_policy_correction.py::test_off_policy_correction -v || python3 off_policy_correction.py
	@echo -e "$(GREEN)✅ Component tests complete$(NC)"

.PHONY: lint
lint: ## Run code quality checks
	@echo -e "$(BLUE)🔍 Running code quality checks...$(NC)"
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "Running flake8..."; \
		flake8 *.py --max-line-length=100 --ignore=E501,W503; \
	else \
		echo -e "$(YELLOW)⚠️  flake8 not found, skipping lint$(NC)"; \
	fi
	@if command -v mypy >/dev/null 2>&1; then \
		echo "Running mypy..."; \
		mypy *.py --ignore-missing-imports; \
	else \
		echo -e "$(YELLOW)⚠️  mypy not found, skipping type checking$(NC)"; \
	fi
	@echo -e "$(GREEN)✅ Code quality checks complete$(NC)"

##@ Docker Build & Deploy
.PHONY: build
build: ## Build Docker image with cache optimization
	@echo -e "$(BLUE)🏗️  Building EfficientZero Docker image...$(NC)"
	@echo "Build configuration:"
	@echo "  Image: $(DOCKER_IMAGE):$(DOCKER_TAG)"
	@echo "  Build Date: $(BUILD_DATE)"
	@echo "  Git Commit: $(GIT_COMMIT)"
	@echo ""
	@DOCKER_BUILDKIT=1 docker build \
		--tag $(DOCKER_IMAGE):$(DOCKER_TAG) \
		--tag $(DOCKER_IMAGE):latest \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		--build-arg VERSION="$(VERSION)" \
		--build-arg GIT_COMMIT="$(GIT_COMMIT)" \
		--cache-from $(DOCKER_IMAGE):latest \
		--file $(DOCKERFILE) \
		.
	@echo -e "$(GREEN)✅ Docker image built successfully$(NC)"

.PHONY: build-no-cache
build-no-cache: ## Build Docker image without cache
	@echo -e "$(BLUE)🏗️  Building EfficientZero Docker image (no cache)...$(NC)"
	@DOCKER_BUILDKIT=1 docker build \
		--no-cache \
		--tag $(DOCKER_IMAGE):$(DOCKER_TAG) \
		--tag $(DOCKER_IMAGE):latest \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		--build-arg VERSION="$(VERSION)" \
		--build-arg GIT_COMMIT="$(GIT_COMMIT)" \
		--file $(DOCKERFILE) \
		.
	@echo -e "$(GREEN)✅ Docker image built successfully (no cache)$(NC)"

.PHONY: push
push: ## Push Docker image to registry
	@if [ -z "$(DOCKER_REGISTRY)" ]; then \
		echo -e "$(RED)❌ DOCKER_REGISTRY not set$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(BLUE)📤 Pushing Docker image to registry...$(NC)"
	@docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@docker tag $(DOCKER_IMAGE):latest $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):latest
	@docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):latest
	@echo -e "$(GREEN)✅ Docker image pushed successfully$(NC)"

##@ Training & Experiments
.PHONY: train
train: ## Start EfficientZero training experiment
	@echo -e "$(BLUE)🚀 Starting EfficientZero training experiment...$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) up -d efficientzero-trainer
	@echo -e "$(GREEN)✅ Training experiment started$(NC)"
	@echo "Monitor with: make logs-trainer"

.PHONY: train-local
train-local: ## Run training locally (without Docker)
	@echo -e "$(BLUE)🚀 Starting local EfficientZero training...$(NC)"
	@python3 efficientzero_main.py --run-experiment --experiment-name local_training
	@echo -e "$(GREEN)✅ Local training complete$(NC)"

.PHONY: experiment
experiment: ## Run custom experiment with parameters
	@echo -e "$(BLUE)🔬 Running custom EfficientZero experiment...$(NC)"
	@read -p "Experiment name: " exp_name; \
	read -p "Episodes (default 10000): " episodes; \
	read -p "Architecture [transformer/tcn/lstm/conv1d] (default transformer): " arch; \
	episodes=$${episodes:-10000}; \
	arch=$${arch:-transformer}; \
	python3 efficientzero_main.py \
		--run-experiment \
		--experiment-name "$$exp_name" \
		--episodes $$episodes \
		--architecture $$arch
	@echo -e "$(GREEN)✅ Custom experiment complete$(NC)"

##@ Container Management  
.PHONY: up
up: ## Start all services
	@echo -e "$(BLUE)🚀 Starting EfficientZero services...$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) up -d
	@echo -e "$(GREEN)✅ Services started$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) ps

.PHONY: down
down: ## Stop all services
	@echo -e "$(BLUE)🛑 Stopping EfficientZero services...$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) down
	@echo -e "$(GREEN)✅ Services stopped$(NC)"

.PHONY: restart
restart: ## Restart all services
	@echo -e "$(BLUE)🔄 Restarting EfficientZero services...$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) restart
	@echo -e "$(GREEN)✅ Services restarted$(NC)"

.PHONY: status
status: ## Show service status
	@echo -e "$(BLUE)📊 EfficientZero service status:$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) ps

##@ Monitoring & Logs
.PHONY: logs
logs: ## Show logs from all services
	@echo -e "$(BLUE)📝 EfficientZero service logs:$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) logs --tail=50 --follow

.PHONY: logs-trainer
logs-trainer: ## Show logs from trainer service
	@echo -e "$(BLUE)📝 EfficientZero trainer logs:$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) logs --tail=100 --follow efficientzero-trainer

.PHONY: logs-monitor
logs-monitor: ## Show logs from monitoring service  
	@echo -e "$(BLUE)📝 EfficientZero monitor logs:$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) logs --tail=100 --follow efficientzero-monitor

.PHONY: dashboard
dashboard: ## Open monitoring dashboard
	@echo -e "$(BLUE)📊 Opening EfficientZero dashboard...$(NC)"
	@echo "Training Dashboard: http://localhost:5003"
	@echo "TensorBoard: http://localhost:6008"  
	@echo "Monitoring Dashboard: http://localhost:8080"
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:8080; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:8080; \
	fi

##@ Development Tools
.PHONY: shell
shell: ## Open shell in trainer container
	@echo -e "$(BLUE)🐚 Opening shell in EfficientZero container...$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) exec efficientzero-trainer /bin/bash

.PHONY: shell-root
shell-root: ## Open root shell in trainer container
	@echo -e "$(BLUE)🐚 Opening root shell in EfficientZero container...$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) exec --user root efficientzero-trainer /bin/bash

.PHONY: debug
debug: ## Run container in debug mode
	@echo -e "$(BLUE)🐛 Starting EfficientZero in debug mode...$(NC)"
	@docker run -it --rm \
		-v "$(PWD):/app/experimental_research" \
		-e DEBUG=1 \
		-e PYTHONPATH=/app/experimental_research \
		$(DOCKER_IMAGE):latest \
		/bin/bash

##@ Data Management
.PHONY: backup
backup: ## Backup results and checkpoints
	@echo -e "$(BLUE)💾 Creating backup of EfficientZero data...$(NC)"
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	backup_dir="backups/efficientzero_$$timestamp"; \
	mkdir -p "$$backup_dir"; \
	cp -r $(RESULTS_DIR) "$$backup_dir/" 2>/dev/null || true; \
	cp -r $(CHECKPOINTS_DIR) "$$backup_dir/" 2>/dev/null || true; \
	cp -r $(LOGS_DIR) "$$backup_dir/" 2>/dev/null || true; \
	echo -e "$(GREEN)✅ Backup created: $$backup_dir$(NC)"

.PHONY: clean-data
clean-data: ## Clean temporary data and logs
	@echo -e "$(YELLOW)⚠️  Cleaning EfficientZero temporary data...$(NC)"
	@read -p "This will delete logs and temporary results. Continue? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf $(LOGS_DIR)/*.log; \
		rm -rf $(RESULTS_DIR)/tmp*; \
		find . -name "*.pyc" -delete; \
		find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true; \
		echo -e "$(GREEN)✅ Temporary data cleaned$(NC)"; \
	else \
		echo "Cancelled"; \
	fi

##@ Cleanup
.PHONY: clean
clean: ## Clean Docker images and containers
	@echo -e "$(BLUE)🧹 Cleaning Docker resources...$(NC)"
	@docker-compose -f $(DEV_COMPOSE_FILE) down --remove-orphans
	@docker system prune -f
	@echo -e "$(GREEN)✅ Docker cleanup complete$(NC)"

.PHONY: clean-all
clean-all: ## Clean everything (images, containers, volumes, data)
	@echo -e "$(RED)⚠️  This will remove ALL Docker resources and data$(NC)"
	@read -p "Continue? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		docker-compose -f $(DEV_COMPOSE_FILE) down -v --remove-orphans; \
		docker system prune -af --volumes; \
		docker image rm $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true; \
		docker image rm $(DOCKER_IMAGE):latest 2>/dev/null || true; \
		rm -rf $(LOGS_DIR) $(RESULTS_DIR) $(CHECKPOINTS_DIR); \
		echo -e "$(GREEN)✅ Complete cleanup finished$(NC)"; \
	else \
		echo "Cancelled"; \
	fi

##@ Quick Commands
.PHONY: quick-start
quick-start: setup build validate train ## Quick start: setup, build, validate, and train
	@echo -e "$(GREEN)🎉 EfficientZero quick start complete!$(NC)"
	@echo "Monitor training: make logs-trainer"
	@echo "View dashboard: make dashboard"

.PHONY: quick-test
quick-test: validate test ## Quick test: validate system and run tests
	@echo -e "$(GREEN)🧪 EfficientZero quick test complete!$(NC)"

##@ Information
.PHONY: info
info: ## Show system information
	@echo -e "$(BLUE)ℹ️  EfficientZero System Information$(NC)"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo "Build Date: $(BUILD_DATE)"  
	@echo "Git Commit: $(GIT_COMMIT)"
	@echo ""
	@echo "Docker Image: $(DOCKER_IMAGE):$(DOCKER_TAG)"
	@echo "Compose File: $(DEV_COMPOSE_FILE)"
	@echo ""
	@echo "Directories:"
	@echo "  Logs: $(LOGS_DIR)"
	@echo "  Results: $(RESULTS_DIR)"
	@echo "  Checkpoints: $(CHECKPOINTS_DIR)"
	@echo "  Data: $(DATA_DIR)"
	@echo "  Configs: $(CONFIGS_DIR)"
	@echo ""
	@if command -v docker >/dev/null 2>&1; then \
		echo "Docker Version: $$(docker --version)"; \
		echo "Docker Compose Version: $$(docker-compose --version)"; \
	fi
	@if command -v python3 >/dev/null 2>&1; then \
		echo "Python Version: $$(python3 --version)"; \
	fi