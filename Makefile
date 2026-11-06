.PHONY: help install test test-cov lint format demo ci docs build clean

help:
	@echo "make install   editable install with dev extras"
	@echo "make test      run the test suite"
	@echo "make test-cov  run the suite with coverage"
	@echo "make lint      ruff"
	@echo "make format    ruff --fix"
	@echo "make demo      offline synthetic demo"
	@echo "make ci        everything CI runs"
	@echo "make docs      build the docs site locally"
	@echo "make build     build sdist and wheel"

install:
	uv pip install -e ".[dev,docs]"

test:
	python3 -m unittest discover -s tests -v

test-cov:
	python3 -m coverage run -m unittest discover -s tests
	python3 -m coverage report

lint:
	ruff check machaon tests examples

format:
	ruff check --fix machaon tests examples

demo:
	python3 -m machaon.cli demo

ci:
	./scripts/ci.sh

docs:
	python3 -m mkdocs serve

build:
	python3 -m build

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf dist build .coverage coverage.xml htmlcov
