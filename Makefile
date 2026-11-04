.PHONY: help install test lint demo ci clean

help:
	@echo "make install   editable install with dev extras"
	@echo "make test      run the test suite"
	@echo "make lint      ruff"
	@echo "make demo      offline synthetic demo"
	@echo "make ci        everything CI runs"

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

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
