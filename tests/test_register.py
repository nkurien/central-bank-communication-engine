from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from pydantic import ValidationError

from cbce.sources.base import ParsedDocument, Release
from cbce.store import raw
from cbce.store.ddl import create_schema
from cbce.store.register import Outcome, register_document
from cbce.store.schema import Bank, DocType

PUBLISHED = datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
FETCHED = PUBLISHED + timedelta(hours=1)
RELEASE = Release(Bank.FED, DocType.STATEMENT, "https://example.test/s", date(2024, 1, 31))


def parsed(published_at: datetime = PUBLISHED) -> ParsedDocument:
    return ParsedDocument("Statement", published_at, "minute", published_at, ("Text.",))


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    create_schema(connection)
    return connection


def count(conn: duckdb.DuckDBPyConnection) -> int:
    return conn.execute("SELECT count(*) FROM documents").fetchone()[0]


def test_new_release_is_stored_and_registered(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # Arrange
    content = b"<html>statement</html>"

    # Act
    document, outcome = register_document(conn, tmp_path, RELEASE, content, parsed(), FETCHED)

    # Assert
    assert outcome is Outcome.NEW
    assert document.doc_id == raw.content_hash(content)
    assert raw.get(tmp_path, document.doc_id) == content
    assert count(conn) == 1


@pytest.mark.parametrize("repeats", [2, 4])
def test_registering_same_content_again_changes_nothing(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path, repeats: int
) -> None:
    # Arrange
    content = b"<html>statement</html>"
    register_document(conn, tmp_path, RELEASE, content, parsed(), FETCHED)

    # Act
    outcomes = [
        register_document(conn, tmp_path, RELEASE, content, parsed(), FETCHED)[1]
        for _ in range(repeats - 1)
    ]

    # Assert
    assert outcomes == [Outcome.UNCHANGED] * (repeats - 1)
    assert count(conn) == 1


def test_changed_content_at_same_url_is_stored_as_new_version_and_flagged(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # Arrange
    register_document(conn, tmp_path, RELEASE, b"original", parsed(), FETCHED)

    # Act
    document, outcome = register_document(conn, tmp_path, RELEASE, b"revised", parsed(), FETCHED)

    # Assert
    assert outcome is Outcome.CHANGED
    assert count(conn) == 2
    assert raw.get(tmp_path, raw.content_hash(b"original")) == b"original"
    assert raw.get(tmp_path, document.doc_id) == b"revised"


@pytest.mark.parametrize(
    "fetched_at",
    [PUBLISHED - timedelta(seconds=1), PUBLISHED - timedelta(days=30)],
    ids=["one_second_early", "a_month_early"],
)
def test_publication_after_fetch_writes_nothing(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path, fetched_at: datetime
) -> None:
    # Arrange: a fetch time that precedes the release time

    # Act / Assert
    with pytest.raises(ValidationError):
        register_document(conn, tmp_path, RELEASE, b"content", parsed(), fetched_at)
    assert count(conn) == 0
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []
