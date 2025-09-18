# Clinic-AI Makefile
# A comprehensive build and development tool for the Clinic-AI project

.PHONY: help install install-dev test test-unit test-integration test-e2e lint format clean docker-build docker-run dev setup bootstrap pre-commit coverage docs

# Default target
help: ## Show this help message
	@echo '🚀 Clinic-AI Development Commands'
	@echo '================================'
	@echo ''
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install production dependencies
	@echo "📦 Installing production dependencies..."
	pip install -e .

install-dev: ## Install development dependencies
	@echo "🔧 Installing development dependencies..."
	pip install -e ".[dev]"

setup: install-dev ## Complete project setup (install + dev deps)
	@echo "✅ Project setup complete!"

bootstrap: ## Run bootstrap script for initial setup
	@echo "🚀 Running bootstrap script..."
	@chmod +x scripts/bootstrap.sh
	./scripts/bootstrap.sh

# Testing targets
test: ## Run all tests
	@echo "🧪 Running all tests..."
	pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	@echo "🧪 Running end-to-end tests..."
	pytest tests/e2e/ -v

test-coverage: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	pytest tests/ --cov=src/clinicai --cov-report=html --cov-report=term-missing

# Code quality targets
lint: ## Run all linting checks
	@echo "🔍 Running linting checks..."
	@echo "  - flake8..."
	flake8 src/ tests/
	@echo "  - black check..."
	black --check src/ tests/
	@echo "  - isort check..."
	isort --check-only src/ tests/
	@echo "  - mypy..."
	mypy src/

format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	black src/ tests/
	isort src/ tests/

pre-commit: ## Run pre-commit checks
	@echo "🔍 Running pre-commit checks..."
	@chmod +x scripts/precommit.sh
	./scripts/precommit.sh

# Development targets
dev: ## Start development server
	@echo "🚀 Starting development server..."
	PYTHONPATH=./src python3 -m uvicorn clinicai.app:app --reload --host 0.0.0.0 --port 8000

dev-debug: ## Start development server with debug logging
	@echo "🚀 Starting development server with debug logging..."
	LOG_LEVEL=DEBUG PYTHONPATH=./src python3 -m uvicorn clinicai.app:app --reload --host 0.0.0.0 --port 8000

# Docker targets
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -f docker/Dockerfile -t clinicai:latest .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p 8000:8000 --env-file .env clinicai:latest

docker-stop: ## Stop running Docker containers
	@echo "🐳 Stopping Docker containers..."
	docker stop $$(docker ps -q --filter ancestor=clinicai:latest) 2>/dev/null || true

docker-clean: ## Clean up Docker images and containers
	@echo "🐳 Cleaning up Docker resources..."
	docker stop $$(docker ps -aq) 2>/dev/null || true
	docker rm $$(docker ps -aq) 2>/dev/null || true
	docker rmi clinicai:latest 2>/dev/null || true

# Documentation targets
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "📚 Serving documentation at http://localhost:8001"
	mkdocs serve -a 127.0.0.1:8001

# Utility targets
clean: ## Clean up generated files and caches
	@echo "🧹 Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/
	rm -rf .coverage.* coverage.xml

clean-all: clean docker-clean ## Clean everything including Docker
	@echo "🧹 Cleaned everything!"

# Database targets
db-start: ## Start MongoDB and Redis (requires Docker)
	@echo "🗄️ Starting databases..."
	docker-compose up -d mongodb redis

db-stop: ## Stop MongoDB and Redis
	@echo "🗄️ Stopping databases..."
	docker-compose down

db-reset: ## Reset databases (WARNING: This will delete all data!)
	@echo "⚠️  Resetting databases..."
	docker-compose down -v
	docker-compose up -d mongodb redis

# Health checks
health: ## Check system health
	@echo "🏥 Checking system health..."
	@echo "  - Python version:"
	@python --version
	@echo "  - Dependencies:"
	@pip list | grep -E "(fastapi|uvicorn|pydantic|motor|beanie|redis)"
	@echo "  - Docker:"
	@docker --version 2>/dev/null || echo "    Docker not installed"
	@echo "  - MongoDB:"
	@docker ps | grep mongodb >/dev/null && echo "    ✅ Running" || echo "    ❌ Not running"
	@echo "  - Redis:"
	@docker ps | grep redis >/dev/null && echo "    ✅ Running" || echo "    ❌ Not running"

# Development workflow
workflow: format lint test ## Run complete development workflow
	@echo "✅ Development workflow completed successfully!"

# Production targets
prod-build: ## Build production-ready application
	@echo "🏭 Building production application..."
	python -m build
	python -m twine check dist/*

prod-install: ## Install production application
	@echo "🏭 Installing production application..."
	pip install dist/*.whl

# Show available commands
commands: ## Show all available commands
	@echo "📋 Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort
