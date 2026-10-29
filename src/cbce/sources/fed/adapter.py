"""Federal Reserve adapter: FOMC statements."""

import re
from datetime import date

from bs4 import BeautifulSoup

from cbce.sources.base import Fetcher, ParsedDocument, Release
from cbce.sources.fed.statement import parse_statement
from cbce.store.schema import Bank, DocType

BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = BASE_URL + "/monetarypolicy/fomchistorical{year}.htm"

# The calendar page covers this year onward; earlier years have one index page each.
CALENDAR_FIRST_YEAR = 2021

# The ordinary meeting statement has one of these titles, sometimes with "and <other action>"
# when the Board announced a discount rate change alongside it. Any other title on a statement
# link (liquidity swaps, which have a ": subtitle", facility announcements, strategy updates)
# is not a meeting statement.
_STATEMENT_TITLE = re.compile(
    r"^(Federal Reserve issues )?FOMC statement( and .+)?$", re.IGNORECASE
)

# A single letter suffix (a, b, c) marks a release of the day; a digit after it (a1) marks the
# implementation note. The statement is usually 'a', but when another release took that slot it
# is 'b' (22 Jan and 16 Dec 2008), so every letter is taken and the page title decides.
# Old pages use /press/monetary/YYYYMMDDa.htm, current ones /pressreleases/monetaryYYYYMMDDa.htm.
_STATEMENT_LINK = re.compile(r"monetary/?(\d{4})(\d{2})(\d{2})([a-z])\.htm$")
# Until about 2005 each release is a folder: /boarddocs/press/monetary/2003/20030129/ (or
# .../general/2000/20000202/ for the earliest years), with the page at default.htm inside.
# These do not redirect to the current scheme, so the link is used as given.
_BOARDDOCS_LINK = re.compile(r"/boarddocs/press/(?:monetary|general)/\d{4}/(\d{4})(\d{2})(\d{2})/")


def statement_releases(index_html: str) -> list[Release]:
    """Statement releases linked from a calendar or yearly index page."""
    found: dict[str, Release] = {}
    for anchor in BeautifulSoup(index_html, "html.parser").find_all("a", href=True):
        href = anchor["href"]
        if match := _BOARDDOCS_LINK.search(href):
            day = date(*map(int, match.groups()[:3]))
            url = f"{BASE_URL}{match[0]}default.htm"
        elif match := _STATEMENT_LINK.search(href):
            day = date(*map(int, match.groups()[:3]))
            # Other old-style URLs redirect to this form, so one scheme serves those years.
            url = f"{BASE_URL}/newsevents/pressreleases/monetary{day:%Y%m%d}{match[4]}.htm"
        else:
            continue
        found[url] = Release(Bank.FED, DocType.STATEMENT, url, day)
    return sorted(found.values(), key=lambda release: (release.release_date, release.url))


def classify(title: str) -> DocType:
    return DocType.STATEMENT if _STATEMENT_TITLE.match(title) else DocType.OTHER_RELEASE


class FedAdapter:
    bank = Bank.FED

    def __init__(self, first_year: int = 1999, last_year: int | None = None) -> None:
        self.first_year = first_year
        self.last_year = last_year

    def discover(self, fetch: Fetcher) -> list[Release]:
        pages = [CALENDAR_URL] + [
            HISTORICAL_URL.format(year=year) for year in range(self.first_year, CALENDAR_FIRST_YEAR)
        ]
        releases: dict[str, Release] = {}
        for page in pages:
            for release in statement_releases(fetch(page).content.decode("utf-8")):
                if release.release_date.year >= self.first_year and (
                    self.last_year is None or release.release_date.year <= self.last_year
                ):
                    releases[release.url] = release
        return sorted(releases.values(), key=lambda release: (release.release_date, release.url))

    def parse(self, release: Release, content: bytes) -> ParsedDocument:
        parsed = parse_statement(content.decode("utf-8"))
        return ParsedDocument(
            title=parsed.title,
            doc_type=classify(parsed.title),
            published_at=parsed.published_at,
            published_at_precision=parsed.published_at_precision,
            event_ts=parsed.published_at,  # the statement is the market event
            paragraphs=parsed.paragraphs,
        )
