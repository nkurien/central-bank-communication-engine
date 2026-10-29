"""Register a fetched release as a Document row. Idempotent and hash-keyed."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

import duckdb

from cbce.sources.base import ParsedDocument, Release
from cbce.store import raw
from cbce.store.schema import Document, Sentence


class Outcome(StrEnum):
    NEW = "new"
    UNCHANGED = "unchanged"  # same bytes already registered
    CHANGED = "changed"  # same URL, different bytes: stored as a new version and flagged


def register_document(
    conn: duckdb.DuckDBPyConnection,
    raw_root: Path,
    release: Release,
    content: bytes,
    parsed: ParsedDocument,
    fetched_at: datetime,
) -> tuple[Document, Outcome]:
    """Store the raw bytes and add a documents row unless this exact content is known.

    The record is validated before anything is written, so a bad timestamp leaves the
    raw store and the table untouched.
    """
    doc_id = raw.content_hash(content)
    existing = conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", [doc_id]).fetchone()
    url_seen = conn.execute(
        "SELECT 1 FROM documents WHERE source_url = ?", [release.url]
    ).fetchone()

    document = Document(
        doc_id=doc_id,
        bank=release.bank,
        doc_type=release.doc_type,
        published_at=parsed.published_at,
        published_at_precision=parsed.published_at_precision,
        event_ts=parsed.event_ts,
        source_url=release.url,
        raw_path=str(raw.raw_path(raw_root, doc_id)),
        fetched_at=fetched_at,
    )
    if existing:
        return document, Outcome.UNCHANGED

    raw.put(raw_root, content)
    conn.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            document.doc_id,
            document.bank.value,
            document.doc_type.value,
            document.published_at,
            document.published_at_precision,
            document.event_ts,
            document.source_url,
            document.raw_path,
            document.fetched_at,
        ],
    )
    return document, Outcome.CHANGED if url_seen else Outcome.NEW


def insert_sentences(conn: duckdb.DuckDBPyConnection, sentences: list[Sentence]) -> None:
    conn.executemany(
        "INSERT INTO sentences VALUES (?, ?, ?, ?, ?)",
        [[s.sentence_id, s.doc_id, s.position, s.text, s.section] for s in sentences],
    )
