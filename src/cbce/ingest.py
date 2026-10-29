"""Run one bank adapter end to end: discover, fetch, parse, segment, register."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from cbce.parse.segment import segment
from cbce.sources.base import Fetcher, SourceAdapter
from cbce.store.register import Outcome, insert_sentences, register_document


@dataclass
class IngestReport:
    new: int = 0
    unchanged: int = 0
    changed: int = 0
    skipped: int = 0
    aborted: bool = False  # stopped early after too many consecutive failures
    failures: list[tuple[str, str]] = field(default_factory=list)  # (url, error)


def ingest(
    adapter: SourceAdapter,
    fetch: Fetcher,
    conn: duckdb.DuckDBPyConnection,
    raw_root: Path,
    *,
    refetch: bool = False,
    limit: int | None = None,
    max_consecutive_failures: int = 3,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> IngestReport:
    """Ingest every release the adapter discovers.

    Idempotent: a URL already registered is skipped unless `refetch` is set, which is how
    changed pages are detected. `limit` keeps only the newest N releases. A release that
    fails to fetch or parse is recorded in the report and writes nothing; the run carries on
    with the next one, but stops after `max_consecutive_failures` in a row so that a block or
    an outage ends the run instead of hammering the site.
    """
    report = IngestReport()
    releases = adapter.discover(fetch)
    consecutive_failures = 0
    for release in releases[len(releases) - limit :] if limit is not None else releases:
        known = conn.execute(
            "SELECT 1 FROM documents WHERE source_url = ?", [release.url]
        ).fetchone()
        if known and not refetch:
            report.skipped += 1
            continue
        try:
            result = fetch(release.url)
            parsed = adapter.parse(release, result.content)
            conn.execute("BEGIN")
            try:
                document, outcome = register_document(
                    conn, raw_root, release, result.content, parsed, now()
                )
                if outcome is not Outcome.UNCHANGED:
                    insert_sentences(conn, segment(document.doc_id, parsed.paragraphs))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        except Exception as error:  # one bad release must not stop the run
            report.failures.append((release.url, f"{type(error).__name__}: {error}"))
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                report.aborted = True
                break
            continue
        consecutive_failures = 0
        match outcome:
            case Outcome.NEW:
                report.new += 1
            case Outcome.CHANGED:
                report.changed += 1
            case Outcome.UNCHANGED:
                report.unchanged += 1
    return report
