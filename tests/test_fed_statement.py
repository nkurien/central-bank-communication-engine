from datetime import UTC, datetime
from pathlib import Path

import pytest

from cbce.sources.fed.statement import ParseError, parse_release_time, parse_statement

FIXTURES = Path(__file__).parent / "fixtures"
STATEMENT_2024_01_31 = (FIXTURES / "fomc_statement_2024-01-31.html").read_text(encoding="utf-8")
STATEMENT_2000_02_02 = (FIXTURES / "fomc_statement_2000-02-02.html").read_text(encoding="utf-8")
STATEMENT_2003_01_29 = (FIXTURES / "fomc_statement_2003-01-29.html").read_text(encoding="utf-8")
STATEMENT_2003_12_09 = (FIXTURES / "fomc_statement_2003-12-09.html").read_text(encoding="utf-8")
STATEMENT_2014_07_30 = (FIXTURES / "fomc_statement_2014-07-30.html").read_text(encoding="utf-8")
STATEMENT_2006_06_29 = (FIXTURES / "fomc_statement_2006-06-29.html").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("date_text", "time_text", "expected_utc"),
    [
        pytest.param(
            "January 31, 2024",
            "For release at 2:00 p.m. EST",
            datetime(2024, 1, 31, 19, 0, tzinfo=UTC),
            id="winter_est",
        ),
        pytest.param(
            "June 12, 2024",
            "For release at 2:00 p.m. EDT",
            datetime(2024, 6, 12, 18, 0, tzinfo=UTC),
            id="summer_edt",
        ),
        pytest.param(
            "June 12, 2024",
            "For release at 2:00 p.m. EST",
            datetime(2024, 6, 12, 18, 0, tzinfo=UTC),
            id="label_ignored_in_summer",
        ),
        pytest.param(
            "March 15, 2020",
            "For release at 5:00 p.m. EDT",
            datetime(2020, 3, 15, 21, 0, tzinfo=UTC),
            id="unscheduled_sunday_meeting",
        ),
        pytest.param(
            "May 4, 1999",
            "For release at 11:30 a.m.",
            datetime(1999, 5, 4, 15, 30, tzinfo=UTC),
            id="morning_no_zone_label",
        ),
    ],
)
def test_parse_release_time_returns_utc_equivalent(
    date_text: str, time_text: str, expected_utc: datetime
) -> None:
    # Arrange: date and time strings as they appear on the page

    # Act
    result, precision = parse_release_time(date_text, time_text)

    # Assert
    assert result == expected_utc
    assert result.tzinfo is not None
    assert precision == "minute"


@pytest.mark.parametrize(
    ("date_text", "time_text", "expected_utc"),
    [
        pytest.param(
            "June 29, 2006",
            "For immediate release",
            datetime(2006, 6, 30, 3, 59, 59, tzinfo=UTC),
            id="summer_edt_end_of_day",
        ),
        pytest.param(
            "February 4, 1999",
            "For immediate release   \n\n Share",
            datetime(1999, 2, 5, 4, 59, 59, tzinfo=UTC),
            id="winter_est_end_of_day",
        ),
    ],
)
def test_parse_release_time_uses_end_of_day_when_only_date_is_known(
    date_text: str, time_text: str, expected_utc: datetime
) -> None:
    # Arrange: older pages say only "For immediate release"

    # Act
    result, precision = parse_release_time(date_text, time_text)

    # Assert
    assert result == expected_utc
    assert precision == "day"


@pytest.mark.parametrize("time_text", ["", "For release at noon", "Immediately"])
def test_parse_release_time_rejects_missing_time(time_text: str) -> None:
    # Arrange: text without a clock time

    # Act / Assert
    with pytest.raises(ParseError):
        parse_release_time("January 31, 2024", time_text)


def test_parse_statement_reads_metadata() -> None:
    # Arrange: saved Fed page for the 31 January 2024 meeting

    # Act
    parsed = parse_statement(STATEMENT_2024_01_31)

    # Assert
    assert parsed.title == "Federal Reserve issues FOMC statement"
    assert parsed.published_at == datetime(2024, 1, 31, 19, 0, tzinfo=UTC)
    assert parsed.published_at_precision == "minute"


def test_parse_statement_handles_older_layout_without_clock_time() -> None:
    # Arrange: saved Fed page for the 29 June 2006 meeting

    # Act
    parsed = parse_statement(STATEMENT_2006_06_29)

    # Assert
    assert parsed.title == "FOMC statement"
    assert parsed.published_at == datetime(2006, 6, 30, 3, 59, 59, tzinfo=UTC)
    assert parsed.published_at_precision == "day"
    assert parsed.paragraphs[0].startswith("The Federal Open Market Committee decided today")
    assert len(parsed.paragraphs) == 4
    assert not any("Voting for" in p or "related action" in p for p in parsed.paragraphs)


@pytest.mark.parametrize(
    ("index", "starts_with"),
    [
        (0, "Recent indicators suggest that economic activity"),
        (1, "The Committee seeks to achieve maximum employment"),
        (2, "In support of its goals"),
        (3, "In assessing the appropriate stance"),
    ],
)
def test_parse_statement_keeps_policy_paragraphs_in_order(index: int, starts_with: str) -> None:
    # Arrange / Act
    parsed = parse_statement(STATEMENT_2024_01_31)

    # Assert
    assert parsed.paragraphs[index].startswith(starts_with)


def test_parse_statement_drops_boilerplate() -> None:
    # Arrange / Act
    parsed = parse_statement(STATEMENT_2024_01_31)

    # Assert
    assert len(parsed.paragraphs) == 4
    assert not any("Voting for" in p or "media inquiries" in p for p in parsed.paragraphs)


@pytest.mark.parametrize(
    "html",
    [
        pytest.param("", id="empty"),
        pytest.param("<html><body><p>Page not Found</p></body></html>", id="not_found_page"),
        pytest.param(
            '<div id="article"><p class="article__time">May 4, 1999</p></div>', id="no_body"
        ),
    ],
)
def test_parse_statement_raises_on_unexpected_structure(html: str) -> None:
    # Arrange: pages that are not a statement

    # Act / Assert
    with pytest.raises(ParseError):
        parse_statement(html)


@pytest.mark.parametrize(
    ("html", "title", "published_utc", "paragraph_count", "last_starts_with"),
    [
        pytest.param(
            STATEMENT_2003_01_29,
            "FOMC statement",
            datetime(2003, 1, 30, 4, 59, 59, tzinfo=UTC),
            3,
            "In these circumstances",
            id="legacy_table_layout_2003",
        ),
        pytest.param(
            STATEMENT_2014_07_30,
            "Federal Reserve issues FOMC statement",
            datetime(2014, 7, 31, 3, 59, 59, tzinfo=UTC),
            6,
            "When the Committee decides",
            id="attachments_panel_layout_2014",
        ),
    ],
)
def test_parse_statement_handles_every_layout_seen_so_far(
    html: str,
    title: str,
    published_utc: datetime,
    paragraph_count: int,
    last_starts_with: str,
) -> None:
    # Arrange: saved pages from layouts other than the 2006 and 2024 ones

    # Act
    parsed = parse_statement(html)

    # Assert
    assert parsed.title == title
    assert parsed.published_at == published_utc
    assert parsed.published_at_precision == "day"
    assert len(parsed.paragraphs) == paragraph_count
    assert parsed.paragraphs[-1].startswith(last_starts_with)


@pytest.mark.parametrize(
    "html",
    [
        STATEMENT_2000_02_02,
        STATEMENT_2003_01_29,
        STATEMENT_2003_12_09,
        STATEMENT_2014_07_30,
        STATEMENT_2006_06_29,
        STATEMENT_2024_01_31,
    ],
    ids=["2000", "2003", "2003_dec", "2014", "2006", "2024"],
)
def test_parsed_paragraphs_contain_no_navigation_votes_or_links(html: str) -> None:
    # Arrange / Act
    paragraphs = parse_statement(html).paragraphs

    # Assert
    junk = ("Voting for", "Statement Regarding", "Home |", "Monetary policy", "Last update")
    assert not [p for p in paragraphs if any(marker in p for marker in junk)]
    assert all(p.endswith((".", "%")) for p in paragraphs)
