.PHONY: setup test lint ingest

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

# Needs CBCE_CONTACT set. Example: make ingest ARGS="--first-year 2021 --limit 10"
ingest:
	uv run python -m cbce.cli ingest $(ARGS)
