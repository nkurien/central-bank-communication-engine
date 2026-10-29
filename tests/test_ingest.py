from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from cbce.ingest import ingest
from cbce.sources.base import FetchResult
from cbce.sources.fed.adapter import CALENDAR_URL, HISTORICAL_URL, FedAdapter
from cbce.store.ddl import create_schema

FIXTURES = Path(__file__).parent / "fixtures"
URL_2024 = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm"
URL_2006 = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20060629a.htm"
PAGE_2024 = (FIXTURES / "fomc_statement_2024-01-31.html").read_bytes()
PAGE_2006 = (FIXTURES / "fomc_statement_2006-06-29.html").read_bytes()
CALENDAR = '<a href="/newsevents/pressreleases/monetary20240131a.htm">HTML</a>'
INDEX_2006 = '<a href="/newsevents/press/monetary/20060629a.htm">Statement</a>'
LATER = datetime(2026, 10, 29, tzinfo=UTC)


def make_fetch(overrides: dict[str, bytes | Exception] | None = None):
    pages: dict[str, bytes | Exception] = {
        CALENDAR_URL: CALENDAR.encode(),
        HISTORICAL_URL.format(year=2006): INDEX_2006.encode(),
        URL_2024: PAGE_2024,
        URL_2006: PAGE_2006,
        **(overrides or {}),
    }

    def fetch(url: str) -> FetchResult:
        page = pages.get(url, b"<html></html>")
        if isinstance(page, Exception):
            raise page
        return FetchResult(page, url)

    return fetch


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    create_schema(connection)
    return connection


def scalar(conn: duckdb.DuckDBPyConnection, sql: str) -> int:
    return conn.execute(sql).fetchone()[0]


def raw_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


def test_ingest_registers_documents_and_sentences(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # Arrange: a calendar listing 2024 and an index listing 2006
    adapter = FedAdapter(first_year=2006)

    # Act
    report = ingest(adapter, make_fetch(), conn, tmp_path, now=lambda: LATER)

    # Assert
    assert (report.new, report.failures) == (2, [])
    assert scalar(conn, "SELECT count(*) FROM documents") == 2
    assert (
        scalar(
            conn,
            "SELECT count(*) FROM sentences s JOIN documents d USING (doc_id)"
            " WHERE d.source_url = '" + URL_2024 + "'",
        )
        == 14
    )
    assert len(raw_files(tmp_path)) == 2


@pytest.mark.parametrize("reruns", [1, 3])
def test_rerun_skips_known_urls_and_changes_nothing(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path, reruns: int
) -> None:
    # Arrange
    adapter = FedAdapter(first_year=2006)
    ingest(adapter, make_fetch(), conn, tmp_path, now=lambda: LATER)

    # Act
    reports = [
        ingest(adapter, make_fetch(), conn, tmp_path, now=lambda: LATER) for _ in range(reruns)
    ]

    # Assert
    assert all(r.skipped == 2 and r.new == 0 for r in reports)
    assert scalar(conn, "SELECT count(*) FROM documents") == 2


@pytest.mark.parametrize(
    ("changed_page", "expected_unchanged", "expected_changed"),
    [
        pytest.param(None, 2, 0, id="same_bytes"),
        pytest.param(PAGE_2024 + b"<!-- edited -->", 2, 0, id="markup_only_edit"),
        pytest.param(PAGE_2024.replace(b"solid pace", b"strong pace"), 1, 1, id="wording_edit"),
    ],
)
def test_refetch_detects_changed_pages(
    conn: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    changed_page: bytes | None,
    expected_unchanged: int,
    expected_changed: int,
) -> None:
    # Arrange
    adapter = FedAdapter(first_year=2006)
    ingest(adapter, make_fetch(), conn, tmp_path, now=lambda: LATER)
    overrides = {URL_2024: changed_page} if changed_page else None

    # Act
    report = ingest(adapter, make_fetch(overrides), conn, tmp_path, refetch=True, now=lambda: LATER)

    # Assert
    assert (report.unchanged, report.changed) == (expected_unchanged, expected_changed)
    assert scalar(conn, "SELECT count(*) FROM documents") == 2 + expected_changed


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(OSError("connection reset"), id="fetch_error"),
        pytest.param(b"<html><body>Page not Found</body></html>", id="not_a_statement"),
    ],
)
def test_failed_release_is_reported_and_the_rest_still_ingest(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path, broken: bytes | Exception
) -> None:
    # Arrange: the 2006 page is broken
    adapter = FedAdapter(first_year=2006)

    # Act
    report = ingest(adapter, make_fetch({URL_2006: broken}), conn, tmp_path, now=lambda: LATER)

    # Assert
    assert report.new == 1
    assert [url for url, _ in report.failures] == [URL_2006]
    assert scalar(conn, "SELECT count(*) FROM documents") == 1
    assert scalar(conn, f"SELECT count(*) FROM documents WHERE source_url = '{URL_2006}'") == 0


def test_release_dated_after_fetch_time_writes_nothing(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # Arrange: a clock earlier than every release, so published_at > fetched_at
    adapter = FedAdapter(first_year=2006)
    early = datetime(2000, 1, 1, tzinfo=UTC)

    # Act
    report = ingest(adapter, make_fetch(), conn, tmp_path, now=lambda: early)

    # Assert
    assert report.new == 0
    assert len(report.failures) == 2
    assert scalar(conn, "SELECT count(*) FROM documents") == 0
    assert scalar(conn, "SELECT count(*) FROM sentences") == 0


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_limit_caps_the_number_of_releases_processed(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path, limit: int
) -> None:
    # Arrange
    adapter = FedAdapter(first_year=2006)

    # Act
    report = ingest(adapter, make_fetch(), conn, tmp_path, limit=limit, now=lambda: LATER)

    # Assert
    assert report.new == limit


def test_limit_keeps_the_newest_releases(conn: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    # Arrange: 2006 is older than 2024
    adapter = FedAdapter(first_year=2006)

    # Act
    ingest(adapter, make_fetch(), conn, tmp_path, limit=1, now=lambda: LATER)

    # Assert
    urls = [row[0] for row in conn.execute("SELECT source_url FROM documents").fetchall()]
    assert urls == [URL_2024]


@pytest.mark.parametrize(
    ("max_failures", "expected_aborted", "expected_failures"),
    [
        pytest.param(1, True, 1, id="stops_after_first_failure"),
        pytest.param(2, True, 2, id="stops_after_two_in_a_row"),
        pytest.param(3, False, 2, id="runs_out_of_releases_first"),
    ],
)
def test_run_aborts_after_consecutive_failures(
    conn: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    max_failures: int,
    expected_aborted: bool,
    expected_failures: int,
) -> None:
    # Arrange: a clock before every release makes each one fail validation
    adapter = FedAdapter(first_year=2006)
    early = datetime(2000, 1, 1, tzinfo=UTC)

    # Act
    report = ingest(
        adapter,
        make_fetch(),
        conn,
        tmp_path,
        max_consecutive_failures=max_failures,
        now=lambda: early,
    )

    # Assert
    assert report.aborted is expected_aborted
    assert len(report.failures) == expected_failures


def test_a_success_resets_the_failure_streak(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    # Arrange: oldest (2006) fails, newest (2024) succeeds, limit of 1 failure in a row
    adapter = FedAdapter(first_year=2006)
    fetch = make_fetch({URL_2006: OSError("reset")})

    # Act
    report = ingest(adapter, fetch, conn, tmp_path, max_consecutive_failures=2, now=lambda: LATER)

    # Assert
    assert (report.new, len(report.failures), report.aborted) == (1, 1, False)
