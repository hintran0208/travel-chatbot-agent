# Makefile for Meeting Transcript Summarizer

.PHONY: help install run clean test lint format check

# Default target
help:
	@echo "Meeting Transcript Summarizer - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install   - Install Python dependencies"
	@echo ""
	@echo "Execution:"
	@echo "  run       - Run the transcript summarizer"
	@echo ""
	@echo "Development:"
	@echo "  lint      - Run code linting with flake8"
	@echo "  format    - Format code with black"
	@echo "  check     - Run all code quality checks"
	@echo "  test      - Run unit tests (if available)"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean     - Clean up generated files and cache"
	@echo "  env-check - Check environment variable configuration"
	@echo ""

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Run the main application
run:
	@echo "🚗 Starting Meeting Transcript Summarize..."
	python main.py

# Environment variable check
env-check:
	@echo "Checking environment configuration..."
	@python -c "import os; from dotenv import load_dotenv; load_dotenv(); \
	vars_to_check = ['OPENAI_API_KEY']; \
	missing = [v for v in vars_to_check if not os.getenv(v)]; \
	print('✅ All required environment variables are set!') if not missing else print(f'❌ Missing variables: {missing}')"

# Code quality checks
lint:
	@echo "Running code linting..."
	@pip list | grep -q flake8 || pip install flake8
	flake8 main.py --max-line-length=120 --ignore=E501,W503

format:
	@echo "Formatting code..."
	@pip list | grep -q black || pip install black
	black main.py --line-length=120

check: lint
	@echo "Running all code quality checks..."
	@python -m py_compile main.py
	@echo "✅ All checks passed!"

# Test runner (placeholder for future tests)
test:
	@echo "Running tests..."
	@python -c "import main; print('✅ Module imports successfully')"
	@echo "✅ Basic smoke tests passed!"

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf summary/
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup completed!"

# Development setup with additional tools
dev-setup: install
	@echo "Setting up development environment..."
	pip install black flake8 pytest
	@echo "✅ Development environment ready!"

# Quick validation of the setup
validate: env-check
	@echo "Validating project setup..."
	@python -c "import openai, dotenv; print('✅ All imports successful')"
	@echo "✅ Project validation completed!"

# Show project information
info:
	@echo "� Meeting Transcript Summarizer"
	@echo "================================"
	@echo "Purpose: Generate concise summaries from meeting transcripts"
	@echo "Technology: OpenAI + Python"
	@echo "Input: Meeting transcript .txt files in transcripts/ folder"
	@echo "Output: Summarized versions in summary/ folder"
	@echo ""
	@echo "Project Structure:"
	@find . -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "Makefile" | head -10
	@echo ""
	@echo "Usage: make run (after configuring .env file)"
