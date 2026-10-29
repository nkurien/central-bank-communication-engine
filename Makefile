.PHONY: setup test lint ingest reparse eval-dev

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

# Needs CBCE_CONTACT set. Example: make ingest ARGS="--first-year 2021 --limit 10"
ingest:
	uv run python -m cbce.cli ingest $(ARGS)

# Re-derive documents and sentences from data/raw. No network.
reparse:
	uv run python -m cbce.cli reparse

# Dev-split table for every baseline, recorded as a run. Refuses on a dirty git tree.
# Examples: make eval-dev    make eval-dev ARGS="--no-record --no-encoder"
eval-dev:
	uv run python -m cbce.cli eval-dev $(ARGS)
