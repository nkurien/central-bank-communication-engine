from pathlib import Path

import duckdb
import pytest

from cbce.cli import build_parser, main
from cbce.sources.base import FetchResult
from cbce.sources.fed.adapter import CALENDAR_URL

FIXTURES = Path(__file__).parent / "fixtures"
PAGE_2024 = (FIXTURES / "fomc_statement_2024-01-31.html").read_bytes()
URL_2024 = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm"
CALENDAR = '<a href="/newsevents/pressreleases/monetary20240131a.htm">HTML</a>'


def fake_fetch(url: str) -> FetchResult:
    pages = {CALENDAR_URL: CALENDAR.encode(), URL_2024: PAGE_2024}
    return FetchResult(pages.get(url, b"<html></html>"), url)


@pytest.mark.parametrize(
    ("argv", "attribute", "expected"),
    [
        pytest.param(["ingest"], "first_year", 1999, id="default_first_year"),
        pytest.param(["ingest"], "interval", 3.0, id="default_interval_is_polite"),
        pytest.param(["ingest", "--limit", "10"], "limit", 10, id="limit"),
        pytest.param(["ingest", "--refetch"], "refetch", True, id="refetch"),
    ],
)
def test_parser_options(argv: list[str], attribute: str, expected: object) -> None:
    # Arrange / Act
    args = build_parser().parse_args(argv)

    # Assert
    assert getattr(args, attribute) == expected


def test_ingest_command_reports_and_stores(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # Arrange
    db, raw = tmp_path / "db" / "cbce.duckdb", tmp_path / "raw"
    argv = ["ingest", "--first-year", "2021", "--db", str(db), "--raw", str(raw)]

    # Act
    code = main(argv, fetch=fake_fetch)

    # Assert
    assert code == 0
    assert "new=1" in capsys.readouterr().out
    conn = duckdb.connect(str(db))
    assert conn.execute("SELECT count(*) FROM sentences").fetchone()[0] == 14
