.PHONY: help install test lint demo ci docs clean

help:
	@echo "make install   editable install with dev extras"
	@echo "make test      run the test suite"
	@echo "make lint      ruff"
	@echo "make demo      offline synthetic demo"
	@echo "make ci        everything CI runs"
	@echo "make docs      build the docs site locally"

install:
	uv pip install -e ".[dev]"

test:
	python3 -m unittest discover -s tests -v

lint:
	ruff check machaon tests examples

demo:
	python3 -m machaon.cli demo

ci:
	./scripts/ci.sh

docs:
	mkdocs serve

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
