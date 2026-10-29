.PHONY: setup test lint

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .
