from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from pydantic import ValidationError

from cbce.sources.base import ParsedDocument, Release
from cbce.store import raw
from cbce.store.ddl import create_schema
from cbce.store.register import Outcome, document_id, register_document
from cbce.store.schema import Bank, DocType

PUBLISHED = datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
FETCHED = PUBLISHED + timedelta(hours=1)
RELEASE = Release(Bank.FED, DocType.STATEMENT, "https://example.test/s", date(2024, 1, 31))


def parsed(
    paragraphs: tuple[str, ...] = ("Text.",),
    doc_type: DocType = DocType.STATEMENT,
    published_at: datetime = PUBLISHED,
) -> ParsedDocument:
    return ParsedDocument("Statement", doc_type, published_at, "minute", published_at, paragraphs)


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
    assert document.doc_id == document_id(parsed())
    assert raw.get(tmp_path, raw.content_hash(content)) == content
    assert Path(document.raw_path) == raw.raw_path(tmp_path, raw.content_hash(content))
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


def test_same_wording_with_different_bytes_is_the_same_document(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # Arrange: e.g. a per-request script injected into the page
    first, _ = register_document(conn, tmp_path, RELEASE, b"page <!-- ray 1 -->", parsed(), FETCHED)

    # Act
    second, outcome = register_document(
        conn, tmp_path, RELEASE, b"page <!-- ray 2 -->", parsed(), FETCHED
    )

    # Assert
    assert outcome is Outcome.UNCHANGED
    assert second.doc_id == first.doc_id
    assert count(conn) == 1


def test_changed_wording_at_same_url_is_stored_as_new_version_and_flagged(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # Arrange
    first, _ = register_document(conn, tmp_path, RELEASE, b"original", parsed(("Old.",)), FETCHED)

    # Act
    second, outcome = register_document(
        conn, tmp_path, RELEASE, b"revised", parsed(("New.",)), FETCHED
    )

    # Assert
    assert outcome is Outcome.CHANGED
    assert second.doc_id != first.doc_id
    assert count(conn) == 2
    assert raw.get(tmp_path, raw.content_hash(b"original")) == b"original"
    assert raw.get(tmp_path, raw.content_hash(b"revised")) == b"revised"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(("A.", "B."), ("A.", "B"), id="punctuation"),
        pytest.param(("A.", "B."), ("B.", "A."), id="order"),
        pytest.param(("A.",), ("A.", "B."), id="extra_paragraph"),
    ],
)
def test_document_id_depends_on_exact_wording(
    first: tuple[str, ...], second: tuple[str, ...]
) -> None:
    # Arrange / Act
    id_first, id_second = document_id(parsed(first)), document_id(parsed(second))

    # Assert
    assert id_first != id_second


@pytest.mark.parametrize("doc_type", [DocType.STATEMENT, DocType.OTHER_RELEASE])
def test_doc_type_comes_from_the_parsed_document(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path, doc_type: DocType
) -> None:
    # Arrange: discovery said statement; the parsed title decides what it really is

    # Act
    document, _ = register_document(
        conn, tmp_path, RELEASE, b"x", parsed(doc_type=doc_type), FETCHED
    )

    # Assert
    assert document.doc_type is doc_type
    stored = conn.execute("SELECT doc_type FROM documents").fetchone()[0]
    assert stored == doc_type.value


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
