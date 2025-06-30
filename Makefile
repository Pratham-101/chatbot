.PHONY: help install install-dev install-prod test test-unit test-integration lint format type-check clean build docker-build docker-run docker-stop deploy

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install base dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

install-prod: ## Install production dependencies
	pip install -e ".[production]"

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-cov: ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run linting checks
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Format code
	black src/ tests/
	isort src/ tests/

type-check: ## Run type checking
	mypy src/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

docker-build: ## Build Docker image
	docker build -t mutual-fund-chatbot:latest .

docker-run: ## Run Docker container
	docker-compose up -d

docker-stop: ## Stop Docker container
	docker-compose down

deploy: ## Deploy to production
	@echo "Deploying to production..."
	# Add your deployment commands here
	# Example: ansible-playbook deploy.yml

setup-dev: install-dev ## Setup development environment
	pre-commit install
	@echo "Development environment setup complete!"

run-dev: ## Run development server
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run production server
	gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

migrate: ## Run database migrations
	@echo "Running migrations..."
	# Add migration commands here

seed: ## Seed database with initial data
	@echo "Seeding database..."
	# Add seed commands here

backup: ## Create database backup
	@echo "Creating backup..."
	# Add backup commands here

monitor: ## Start monitoring
	@echo "Starting monitoring..."
	# Add monitoring commands here

logs: ## View application logs
	docker-compose logs -f

health: ## Check application health
	curl -f http://localhost:8000/health || echo "Application is not healthy" 