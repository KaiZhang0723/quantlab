# Convenience targets for local development.

.PHONY: install test lint cov docs clean reports

install:
	pip install -e ".[dev,docs]"

test:
	pytest -q

cov:
	pytest --cov=quantlab --cov-branch --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests

docs:
	sphinx-build -b html docs docs/_build/html

reports:
	python scripts/generate_reports.py

clean:
	rm -rf .pytest_cache .coverage htmlcov coverage.xml .mypy_cache .ruff_cache .hypothesis
	find . -name "__pycache__" -type d -exec rm -rf {} +
