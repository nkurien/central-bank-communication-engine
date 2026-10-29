from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from cbce.sources.base import FetchResult, Release
from cbce.sources.fed.adapter import (
    CALENDAR_URL,
    HISTORICAL_URL,
    FedAdapter,
    classify,
    statement_releases,
)
from cbce.store.schema import Bank, DocType

FIXTURES = Path(__file__).parent / "fixtures"
CALENDAR = (FIXTURES / "fomc_calendar.html").read_text(encoding="utf-8")
HISTORICAL_2000 = (FIXTURES / "fomc_historical_2000.html").read_text(encoding="utf-8")
HISTORICAL_2003 = (FIXTURES / "fomc_historical_2003.html").read_text(encoding="utf-8")
HISTORICAL_2006 = (FIXTURES / "fomc_historical_2006.html").read_text(encoding="utf-8")
STATEMENT_2024 = (FIXTURES / "fomc_statement_2024-01-31.html").read_bytes()


def fake_fetcher(pages: dict[str, str]):
    """Return a fetcher serving `pages`, an empty page otherwise, and a log of requested URLs."""
    requested: list[str] = []

    def fetch(url: str) -> FetchResult:
        requested.append(url)
        return FetchResult(pages.get(url, "<html></html>").encode(), url)

    fetch.requested = requested  # type: ignore[attr-defined]
    return fetch


@pytest.mark.parametrize(
    ("day", "expected_in_result"),
    [
        pytest.param(date(2024, 1, 31), True, id="statement_listed"),
        pytest.param(date(2021, 1, 27), True, id="first_calendar_year"),
        pytest.param(date(2026, 1, 28), True, id="latest_year_listed"),
        pytest.param(date(2024, 1, 30), False, id="day_before_meeting"),
    ],
)
def test_calendar_statements_include_only_real_meeting_dates(
    day: date, expected_in_result: bool
) -> None:
    # Arrange: saved FOMC calendar page

    # Act
    days = {release.release_date for release in statement_releases(CALENDAR)}

    # Assert
    assert (day in days) is expected_in_result


def test_calendar_statements_exclude_notes_and_other_releases() -> None:
    # Arrange: the page also links implementation notes (a1), strategy statements (b), others

    # Act
    releases = statement_releases(CALENDAR)

    # Assert
    assert all(release.url.endswith("a.htm") for release in releases)
    assert not any("a1.htm" in release.url or "b.htm" in release.url for release in releases)
    assert len({release.release_date for release in releases}) == len(releases)


def test_calendar_statements_are_sorted_oldest_first() -> None:
    # Arrange / Act
    days = [release.release_date for release in statement_releases(CALENDAR)]

    # Assert
    assert days == sorted(days)


@pytest.mark.parametrize(
    "day",
    [
        date(2006, 1, 31),
        date(2006, 3, 28),
        date(2006, 5, 10),
        date(2006, 6, 29),
        date(2006, 8, 8),
        date(2006, 9, 20),
        date(2006, 10, 25),
        date(2006, 12, 12),
    ],
    ids=lambda d: d.isoformat(),
)
def test_yearly_index_yields_each_2006_meeting_with_current_url_scheme(day: date) -> None:
    # Arrange: saved 2006 index, which uses the old /press/monetary/ links

    # Act
    releases = {release.release_date: release for release in statement_releases(HISTORICAL_2006)}

    # Assert
    expected = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{day:%Y%m%d}a.htm"
    assert releases[day] == Release(Bank.FED, DocType.STATEMENT, expected, day)


@pytest.mark.parametrize("html", ["", "<html></html>", '<a href="/about.htm">About</a>'])
def test_pages_without_statement_links_yield_nothing(html: str) -> None:
    # Arrange: pages with no statement links

    # Act
    result = statement_releases(html)

    # Assert
    assert result == []


@pytest.mark.parametrize(
    ("first_year", "last_year", "expected_years"),
    [
        pytest.param(2006, 2006, {2006}, id="single_year"),
        pytest.param(2006, 2021, {2006, 2021}, id="across_index_and_calendar"),
    ],
)
def test_discover_merges_index_pages_and_respects_year_bounds(
    first_year: int, last_year: int, expected_years: set[int]
) -> None:
    # Arrange
    fetch = fake_fetcher(
        {CALENDAR_URL: CALENDAR, HISTORICAL_URL.format(year=2006): HISTORICAL_2006}
    )
    adapter = FedAdapter(first_year=first_year, last_year=last_year)

    # Act
    releases = adapter.discover(fetch)

    # Assert
    days = [release.release_date for release in releases]
    assert {day.year for day in days} == expected_years
    assert days == sorted(days)


def test_discover_fetches_calendar_and_each_earlier_year_once() -> None:
    # Arrange
    fetch = fake_fetcher({})
    adapter = FedAdapter(first_year=2019)

    # Act
    adapter.discover(fetch)

    # Assert
    assert fetch.requested == [
        CALENDAR_URL,
        HISTORICAL_URL.format(year=2019),
        HISTORICAL_URL.format(year=2020),
    ]


def test_parse_marks_statement_as_the_market_event() -> None:
    # Arrange
    release = Release(Bank.FED, DocType.STATEMENT, "https://example.test/s", date(2024, 1, 31))

    # Act
    parsed = FedAdapter().parse(release, STATEMENT_2024)

    # Assert
    assert parsed.published_at == datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
    assert parsed.event_ts == parsed.published_at
    assert len(parsed.paragraphs) == 4


@pytest.mark.parametrize(
    "day",
    [date(2003, 1, 29), date(2003, 3, 18), date(2003, 8, 12), date(2003, 12, 9)],
    ids=lambda d: d.isoformat(),
)
def test_oldest_index_uses_folder_links_kept_as_given(day: date) -> None:
    # Arrange: saved 2003 index, whose links are /boarddocs/press/monetary/YYYY/YYYYMMDD/

    # Act
    releases = {r.release_date: r for r in statement_releases(HISTORICAL_2003)}

    # Assert
    expected = f"https://www.federalreserve.gov/boarddocs/press/monetary/{day.year}/{day:%Y%m%d}/default.htm"
    assert releases[day].url == expected


def test_oldest_index_yields_the_eight_scheduled_meetings() -> None:
    # Arrange / Act
    releases = statement_releases(HISTORICAL_2003)

    # Assert
    assert len(releases) == 8
    assert {r.release_date.year for r in releases} == {2003}


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        pytest.param("FOMC statement", DocType.STATEMENT, id="older_title"),
        pytest.param("Federal Reserve issues FOMC statement", DocType.STATEMENT, id="newer_title"),
        pytest.param(
            "FOMC statement: The Federal Reserve is providing liquidity",
            DocType.OTHER_RELEASE,
            id="subtitle_means_not_a_meeting_statement",
        ),
        pytest.param(
            "Statement Regarding Monetary Policy Implementation",
            DocType.OTHER_RELEASE,
            id="implementation_statement",
        ),
        pytest.param(
            "FOMC statement and Board discount rate action",
            DocType.STATEMENT,
            id="combined_with_discount_rate_action",
        ),
        pytest.param(
            "FOMC statement: Federal Reserve and other central banks announce measures",
            DocType.OTHER_RELEASE,
            id="colon_subtitle_with_and_later_on",
        ),
        pytest.param("", DocType.OTHER_RELEASE, id="empty_title"),
    ],
)
def test_classify_tags_only_ordinary_statement_titles(title: str, expected: DocType) -> None:
    # Arrange: title as parsed from the page

    # Act
    result = classify(title)

    # Assert
    assert result is expected


@pytest.mark.parametrize(
    "day",
    [date(2000, 2, 2), date(2000, 6, 28), date(2000, 12, 19)],
    ids=lambda d: d.isoformat(),
)
def test_earliest_index_uses_general_folder_links(day: date) -> None:
    # Arrange: saved 2000 index, whose links are /boarddocs/press/general/YYYY/YYYYMMDD/

    # Act
    releases = {r.release_date: r for r in statement_releases(HISTORICAL_2000)}

    # Assert
    assert releases[day].url == (
        f"https://www.federalreserve.gov/boarddocs/press/general/{day.year}/{day:%Y%m%d}/default.htm"
    )
    assert len(releases) == 8
