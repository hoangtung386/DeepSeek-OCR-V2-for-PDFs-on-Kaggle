.PHONY: install run test lint clean

install:
	pip install -e ".[dev]"

run:
	python -m src.cli --data-dir data --output-dir output

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/
