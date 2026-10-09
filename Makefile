.PHONY: install lint typecheck test test-postgres eval build scan check clean

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

typecheck:
	python -m mypy

test:
	pytest -m "not live"

test-postgres:
	pytest -m postgres

eval:
	saferag eval --output reports

build:
	python -m build

scan:
	python scripts/safety_scan.py

check: lint typecheck test eval build scan

clean:
	rm -rf build dist reports .pytest_cache .mypy_cache .ruff_cache src/*.egg-info
