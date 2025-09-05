# Keynote MCP Server - Development Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test test-cov lint format type-check clean run-server run-client demo docs

# Default target
help:
	@echo "🎯 Keynote MCP Server - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  run-server   Start the MCP server"
	@echo "  run-client   Start the AI client"
	@echo "  demo         Run the demo suite"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  test         Run all tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black"
	@echo "  type-check   Run type checking with mypy"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Generate documentation"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean        Clean up temporary files"

# Installation commands
install:
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "🔧 Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov black flake8 mypy

# Development commands
run-server:
	@echo "🚀 Starting MCP server..."
	python src/keynote_mcp_server.py

run-client:
	@echo "🤖 Starting AI client..."
	python src/ai_client.py

demo:
	@echo "🎬 Running demo suite..."
	python examples/demo_usage.py

# Testing commands
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

test-cov:
	@echo "📊 Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality commands
lint:
	@echo "🔍 Running linting checks..."
	flake8 src/ tests/ examples/
	@echo "✅ Linting complete"

format:
	@echo "🎨 Formatting code..."
	black src/ tests/ examples/
	@echo "✅ Code formatted"

type-check:
	@echo "🔍 Running type checks..."
	mypy src/
	@echo "✅ Type checking complete"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "Documentation generation not yet implemented"
	@echo "TODO: Add Sphinx configuration"

# Maintenance
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	@echo "✅ Cleanup complete"

# Development workflow
dev-setup: install-dev
	@echo "🔧 Setting up development environment..."
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file from template..."; \
		cp config.env.example .env; \
		echo "⚠️  Please edit .env file with your API keys"; \
	fi
	@echo "✅ Development environment ready"

# Quick development cycle
dev-test: format lint type-check test
	@echo "✅ All quality checks passed"

# Production deployment
deploy-check: dev-test
	@echo "🚀 Production deployment checks passed"
	@echo "Ready for deployment!"

# Help for specific targets
install-help:
	@echo "Installation Help:"
	@echo "1. Create virtual environment: python -m venv venv"
	@echo "2. Activate virtual environment: source venv/bin/activate"
	@echo "3. Install dependencies: make install-dev"
	@echo "4. Configure environment: make dev-setup"

run-help:
	@echo "Running Help:"
	@echo "1. Start MCP server: make run-server"
	@echo "2. In another terminal, start AI client: make run-client"
	@echo "3. Or run the demo: make demo"

test-help:
	@echo "Testing Help:"
	@echo "1. Run all tests: make test"
	@echo "2. Run with coverage: make test-cov"
	@echo "3. Run specific test: pytest tests/test_mathematical_tools.py -v"