"""Re-derive documents and sentences from the stored raw files. No network."""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

from cbce.parse.segment import segment
from cbce.sources.base import ParsedDocument, Release, SourceAdapter
from cbce.store.register import insert_sentences, register_document
from cbce.store.schema import Bank, DocType

# Tables whose rows point at documents or sentences. Reparsing can change doc_ids, which
# would orphan them, so it refuses to run while any of them holds rows.
_DEPENDENTS = ("stance_scores", "doc_scores", "events")

_FETCHED_AT_UTC = "strftime(fetched_at AT TIME ZONE 'UTC', '%Y-%m-%d %H:%M:%S.%f')"


@dataclass
class ReparseReport:
    documents: int = 0
    reclassified: int = 0  # doc_type changed
    reidentified: int = 0  # doc_id changed because the parsed wording changed


def reparse(adapter: SourceAdapter, conn: duckdb.DuckDBPyConnection) -> ReparseReport:
    """Rebuild the documents and sentences tables from the raw files they point at.

    Every page is parsed before anything is deleted, so a parse failure leaves the store
    exactly as it was. DuckDB checks foreign keys against committed rows only, so sentences
    cannot be deleted in the same transaction as their documents: they are cleared first,
    then documents and sentences are rebuilt in one transaction. A database error between
    those steps leaves documents without sentences; running reparse again repairs it.
    """
    for table in _DEPENDENTS:
        if conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]:
            raise RuntimeError(f"{table} has rows that depend on the documents; not reparsing")

    rows = conn.execute(
        f"SELECT doc_id, doc_type, source_url, raw_path, {_FETCHED_AT_UTC} "
        "FROM documents ORDER BY published_at, source_url"
    ).fetchall()
    staged: list[tuple[str, str, Release, bytes, ParsedDocument, datetime]] = []
    for doc_id, doc_type, url, raw_path, fetched_text in rows:
        content = Path(raw_path).read_bytes()
        # parse() does not use the release date, so a placeholder stands in for it.
        release = Release(Bank.FED, DocType.STATEMENT, url, date.min)
        fetched_at = datetime.strptime(fetched_text, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=UTC)
        staged.append(
            (doc_id, doc_type, release, content, adapter.parse(release, content), fetched_at)
        )

    report = ReparseReport(documents=len(staged))
    raw_root = Path(rows[0][3]).parents[1] if rows else Path()
    conn.execute("DELETE FROM sentences")
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM documents")
        for old_id, old_type, release, content, parsed, fetched_at in staged:
            document, _ = register_document(conn, raw_root, release, content, parsed, fetched_at)
            insert_sentences(conn, segment(document.doc_id, parsed.paragraphs))
            report.reclassified += document.doc_type.value != old_type
            report.reidentified += document.doc_id != old_id
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return report
