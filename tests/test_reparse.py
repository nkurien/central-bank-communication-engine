from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from cbce.ingest import ingest
from cbce.reparse import reparse
from cbce.sources.base import FetchResult, ParsedDocument, Release
from cbce.sources.fed.adapter import CALENDAR_URL, FedAdapter
from cbce.store.ddl import create_schema
from cbce.store.schema import DocType

FIXTURES = Path(__file__).parent / "fixtures"
PAGE_2024 = (FIXTURES / "fomc_statement_2024-01-31.html").read_bytes()
URL_2024 = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm"
CALENDAR = '<a href="/newsevents/pressreleases/monetary20240131a.htm">HTML</a>'
LATER = datetime(2026, 10, 29, tzinfo=UTC)


def fake_fetch(url: str) -> FetchResult:
    pages = {CALENDAR_URL: CALENDAR.encode(), URL_2024: PAGE_2024}
    return FetchResult(pages.get(url, b"<html></html>"), url)


class StaleAdapter(FedAdapter):
    """Mimics an earlier, stricter classification rule that mislabelled meeting statements."""

    def __init__(self, wrong_type: DocType) -> None:
        super().__init__(first_year=2021)
        self.wrong_type = wrong_type

    def parse(self, release: Release, content: bytes) -> ParsedDocument:
        return replace(super().parse(release, content), doc_type=self.wrong_type)


def build_store(tmp_path: Path, adapter: FedAdapter) -> tuple[duckdb.DuckDBPyConnection, Path]:
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    ingest(adapter, fake_fetch, conn, tmp_path / "raw", now=lambda: LATER)
    return conn, tmp_path / "raw"


@pytest.fixture
def populated(tmp_path: Path) -> tuple[duckdb.DuckDBPyConnection, Path]:
    return build_store(tmp_path, FedAdapter(first_year=2021))


def rows(conn: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
    return conn.execute(sql).fetchall()


def test_reparse_of_unchanged_store_changes_nothing(
    populated: tuple[duckdb.DuckDBPyConnection, Path],
) -> None:
    # Arrange
    conn, _ = populated
    before = rows(conn, "SELECT doc_id, doc_type FROM documents")
    sentences_before = rows(conn, "SELECT sentence_id FROM sentences ORDER BY 1")

    # Act
    report = reparse(FedAdapter(), conn)

    # Assert
    assert (report.documents, report.reclassified, report.reidentified) == (1, 0, 0)
    assert rows(conn, "SELECT doc_id, doc_type FROM documents") == before
    assert rows(conn, "SELECT sentence_id FROM sentences ORDER BY 1") == sentences_before


@pytest.mark.parametrize("stale_type", ["other_release", "speech"])
def test_reparse_repairs_a_stale_classification(tmp_path: Path, stale_type: str) -> None:
    # Arrange: a store built under an earlier rule that mislabelled the meeting statement
    conn, _ = build_store(tmp_path, StaleAdapter(DocType(stale_type)))
    assert rows(conn, "SELECT doc_type FROM documents") == [(stale_type,)]

    # Act
    report = reparse(FedAdapter(), conn)

    # Assert
    assert report.reclassified == 1
    assert rows(conn, "SELECT doc_type FROM documents") == [("statement",)]


def test_reparse_never_touches_raw_files(
    populated: tuple[duckdb.DuckDBPyConnection, Path],
) -> None:
    # Arrange
    conn, raw_root = populated
    files = {path: path.read_bytes() for path in raw_root.rglob("*") if path.is_file()}

    # Act
    reparse(FedAdapter(), conn)

    # Assert
    assert {path: path.read_bytes() for path in raw_root.rglob("*") if path.is_file()} == files


@pytest.mark.parametrize("table", ["stance_scores", "doc_scores", "events"])
def test_reparse_refuses_when_scores_depend_on_the_documents(
    populated: tuple[duckdb.DuckDBPyConnection, Path], table: str
) -> None:
    # Arrange: a row that depends on the stored documents exists
    conn, _ = populated
    doc_id = rows(conn, "SELECT doc_id FROM documents")[0][0]
    inserts = {
        "stance_scores": (
            "INSERT INTO stance_scores "
            "SELECT sentence_id, 'x', 'other', 'neutral', 0.5, NULL, 'r' FROM sentences LIMIT 1"
        ),
        "doc_scores": f"INSERT INTO doc_scores VALUES ('{doc_id}', 'x', NULL, 0.0, NULL, 'r')",
        "events": (
            f"INSERT INTO events VALUES ('e', 'fed', '{doc_id}', '2024-01-31 19:00+00', "
            "'2024-01-30', '2024-01-31', NULL, NULL)"
        ),
    }
    conn.execute(inserts[table])

    # Act / Assert
    with pytest.raises(RuntimeError, match=table):
        reparse(FedAdapter(), conn)
    assert rows(conn, "SELECT count(*) FROM documents") == [(1,)]


def test_reparse_failure_leaves_the_store_untouched(
    populated: tuple[duckdb.DuckDBPyConnection, Path],
) -> None:
    # Arrange: the raw file now holds something the parser rejects
    conn, _ = populated
    raw_path = Path(rows(conn, "SELECT raw_path FROM documents")[0][0])
    raw_path.chmod(0o644)
    raw_path.write_bytes(b"<html><body>Page not Found</body></html>")
    before = rows(conn, "SELECT doc_id FROM documents")

    # Act / Assert
    with pytest.raises(Exception):  # noqa: B017  any parse error must abort before deleting
        reparse(FedAdapter(), conn)
    assert rows(conn, "SELECT doc_id FROM documents") == before
    assert rows(conn, "SELECT count(*) FROM sentences") == [(14,)]
