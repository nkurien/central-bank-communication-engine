from datetime import UTC, datetime, timedelta

import duckdb
import pytest
from pydantic import ValidationError

from cbce.store.ddl import TABLES, create_schema
from cbce.store.schema import Bank, DocType, Document, Stance, StanceScore, Topic

PUBLISHED = datetime(2024, 1, 31, 19, 0, tzinfo=UTC)


def make_document(**overrides: object) -> Document:
    fields: dict[str, object] = {
        "doc_id": "abc",
        "bank": Bank.FED,
        "doc_type": DocType.STATEMENT,
        "published_at": PUBLISHED,
        "event_ts": PUBLISHED,
        "source_url": "https://example.test/statement",
        "raw_path": "data/raw/abc",
        "fetched_at": PUBLISHED + timedelta(days=1),
    }
    return Document(**{**fields, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="statement_at_release"),
        pytest.param(
            {"doc_type": DocType.PRESS_CONFERENCE, "event_ts": PUBLISHED - timedelta(days=2)},
            id="press_conference_event_precedes_transcript",
        ),
        pytest.param({"fetched_at": PUBLISHED}, id="fetched_at_the_moment_of_release"),
    ],
)
def test_document_accepts_valid_timestamps(overrides: dict[str, object]) -> None:
    # Arrange: overrides applied to a valid document

    # Act
    doc = make_document(**overrides)

    # Assert
    assert doc.event_ts <= doc.published_at <= doc.fetched_at


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"fetched_at": PUBLISHED - timedelta(seconds=1)}, id="published_after_fetch"),
        pytest.param({"event_ts": PUBLISHED + timedelta(hours=1)}, id="event_after_publication"),
        pytest.param({"published_at": datetime(2024, 1, 31, 19, 0)}, id="naive_published_at"),
        pytest.param({"event_ts": datetime(2024, 1, 31, 19, 0)}, id="naive_event_ts"),
        pytest.param({"bank": "snb"}, id="unknown_bank"),
        pytest.param({"unexpected": 1}, id="extra_field"),
    ],
)
def test_document_rejects_invalid_records(overrides: dict[str, object]) -> None:
    # Arrange: overrides applied to a valid document

    # Act / Assert
    with pytest.raises(ValidationError):
        make_document(**overrides)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 5.0])
def test_stance_score_rejects_confidence_outside_unit_interval(confidence: float) -> None:
    # Arrange
    fields = {
        "sentence_id": "s1",
        "scorer": "llm",
        "topic": Topic.INFLATION,
        "stance": Stance.HAWKISH,
        "confidence": confidence,
        "run_id": "r1",
    }

    # Act / Assert
    with pytest.raises(ValidationError):
        StanceScore(**fields)


@pytest.mark.parametrize("repeats", [1, 2, 3])
def test_create_schema_is_idempotent(repeats: int) -> None:
    # Arrange
    conn = duckdb.connect(":memory:")

    # Act
    for _ in range(repeats):
        create_schema(conn)

    # Assert
    found = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert found == set(TABLES)


@pytest.mark.parametrize(
    ("published", "event", "fetched"),
    [
        pytest.param(
            "2024-01-31 19:00+00",
            "2024-01-31 19:00+00",
            "2024-01-30 19:00+00",
            id="published_after_fetch",
        ),
        pytest.param(
            "2024-01-31 19:00+00",
            "2024-02-01 19:00+00",
            "2024-02-02 19:00+00",
            id="event_after_publication",
        ),
    ],
)
def test_database_rejects_inconsistent_timestamps(published: str, event: str, fetched: str) -> None:
    # Arrange
    conn = duckdb.connect(":memory:")
    create_schema(conn)

    # Act / Assert
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "INSERT INTO documents VALUES ('d', 'fed', 'statement', ?, 'minute', ?, 'u', 'p', ?)",
            [published, event, fetched],
        )
