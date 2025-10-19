# Makefile for A2A-ACP Server

.PHONY: help install dev-install format lint type test quality clean

help:
	@echo "Available commands:"
	@echo "  install      - Create venv and install production dependencies."
	@echo "  dev-install  - Create venv and install all (prod + dev) dependencies."
	@echo "  format       - Format code with black and ruff."
	@echo "  lint         - Lint code with ruff."
	@echo "  type         - Type-check code with mypy."
	@echo "  test         - Run tests with pytest."
	@echo "  quality      - Run all quality checks (format, lint, type, test)."
	@echo "  clean        - Remove virtual environment and cache files."

# Create a virtual environment if it doesn't exist
.venv:
	python3 -m venv .venv
	./.venv/bin/pip install -U pip uv

# Installation targets
install: .venv
	@echo "--> Installing production dependencies..."
	./.venv/bin/uv pip install .

dev-install: .venv
	@echo "--> Installing development dependencies..."
	./.venv/bin/uv pip install ".[dev]"

# Quality check targets, depend on dev-install to ensure tools are present
format: dev-install
	@echo "--> Formatting code..."
	./.venv/bin/uv run black src tests
	./.venv/bin/uv run ruff format src tests
	./.venv/bin/uv run ruff check src tests --fix

lint: dev-install
	@echo "--> Linting code..."
	./.venv/bin/uv run ruff check src tests

type: dev-install
	@echo "--> Type-checking code..."
	./.venv/bin/uv run mypy src

test: dev-install
	@echo "--> Running tests..."
	timeout 60 ./.venv/bin/uv run pytest

quality: format lint type test

# Cleanup target
clean:
	@echo "--> Cleaning up..."
	rm -rf .venv
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -r {} +

run: dev-install
	@echo "--> Starting A2A-ACP Server..."
	./.venv/bin/uv run uvicorn src.a2a_acp.main:create_app --port $(or $(PORT),8001) --host "0.0.0.0"