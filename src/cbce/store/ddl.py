"""DuckDB tables mirroring the records in schema.py."""

import duckdb

DDL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR PRIMARY KEY,
    bank VARCHAR NOT NULL,
    doc_type VARCHAR NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    published_at_precision VARCHAR NOT NULL CHECK (published_at_precision IN ('minute', 'day')),
    event_ts TIMESTAMPTZ NOT NULL,
    source_url VARCHAR NOT NULL,
    raw_path VARCHAR NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    CHECK (event_ts <= published_at),
    CHECK (published_at <= fetched_at)
);
CREATE TABLE IF NOT EXISTS sentences (
    sentence_id VARCHAR PRIMARY KEY,
    doc_id VARCHAR NOT NULL REFERENCES documents (doc_id),
    position INTEGER NOT NULL,
    text VARCHAR NOT NULL,
    section VARCHAR
);
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id VARCHAR PRIMARY KEY,
    model VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    prompt_version VARCHAR NOT NULL,
    raw_response VARCHAR NOT NULL,
    tokens INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS stance_scores (
    sentence_id VARCHAR NOT NULL REFERENCES sentences (sentence_id),
    scorer VARCHAR NOT NULL,
    topic VARCHAR NOT NULL,
    stance VARCHAR NOT NULL,
    confidence DOUBLE NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    evidence VARCHAR,
    run_id VARCHAR NOT NULL,
    PRIMARY KEY (sentence_id, scorer, run_id)
);
CREATE TABLE IF NOT EXISTS doc_scores (
    doc_id VARCHAR NOT NULL REFERENCES documents (doc_id),
    scorer VARCHAR NOT NULL,
    topic VARCHAR,
    score DOUBLE NOT NULL,
    diff_vs_previous DOUBLE,
    run_id VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS yield_obs (
    series VARCHAR NOT NULL,
    date DATE NOT NULL,
    value DOUBLE NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (series, date)
);
CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR PRIMARY KEY,
    bank VARCHAR NOT NULL,
    doc_id VARCHAR NOT NULL REFERENCES documents (doc_id),
    release_ts TIMESTAMPTZ NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    yield_change_2y DOUBLE,
    yield_change_10y DOUBLE,
    CHECK (window_start < window_end)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    git_sha VARCHAR NOT NULL,
    config_hash VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    metrics_json VARCHAR NOT NULL
);
"""

TABLES = (
    "documents",
    "sentences",
    "llm_calls",
    "stance_scores",
    "doc_scores",
    "yield_obs",
    "events",
    "runs",
)


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables. Safe to run repeatedly."""
    conn.execute(DDL)
