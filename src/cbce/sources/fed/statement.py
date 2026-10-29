"""Parse a saved Federal Reserve FOMC statement page."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

NEW_YORK = ZoneInfo("America/New_York")

# Paragraphs that follow the policy text and carry no stance.
_BOILERPLATE = re.compile(
    r"^(Voting for|Voting against|For media inquiries|Implementation Note|In a related action)",
    re.IGNORECASE,
)
_RELEASE_TIME = re.compile(r"(\d{1,2}):(\d{2})\s*([ap])\.m\.", re.IGNORECASE)
_IMMEDIATE = re.compile(r"immediate release", re.IGNORECASE)

Precision = Literal["minute", "day"]


class ParseError(ValueError):
    """The page does not have the expected structure. Never write partial rows."""


@dataclass(frozen=True)
class ParsedStatement:
    title: str
    published_at: datetime
    published_at_precision: Precision
    paragraphs: tuple[str, ...]


def parse_release_time(date_text: str, time_text: str) -> tuple[datetime, Precision]:
    """Combine 'January 31, 2024' and 'For release at 2:00 p.m. EST' into an aware datetime.

    The EST/EDT label is ignored: the clock time is read as America/New_York local time,
    which resolves daylight saving from the date. Older pages say only 'For immediate
    release'; those get 23:59:59 ET on the release date and precision 'day', so the
    timestamp is never earlier than the true release.
    """
    day = datetime.strptime(date_text.strip(), "%B %d, %Y")
    match = _RELEASE_TIME.search(time_text)
    if match is None:
        if _IMMEDIATE.search(time_text):
            return day.replace(hour=23, minute=59, second=59, tzinfo=NEW_YORK), "day"
        raise ParseError(f"no release time in {time_text!r}")
    hour, minute, meridiem = int(match[1]), int(match[2]), match[3].lower()
    hour = hour % 12 + (12 if meridiem == "p" else 0)
    return day.replace(hour=hour, minute=minute, tzinfo=NEW_YORK), "minute"


def parse_statement(html: str) -> ParsedStatement:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find(id="article")
    date = article and article.find(class_="article__time")
    title = article and article.find(class_="title")
    release = article and article.find(class_="releaseTime")
    if not (article and date and title and release):
        raise ParseError("missing article, date, title or release time")

    body = article.find_all("div", class_="col-md-8")[-1]
    paragraphs = tuple(
        text
        for p in body.find_all("p")
        if (text := " ".join(p.get_text().split())) and not _BOILERPLATE.match(text)
    )
    if not paragraphs:
        raise ParseError("no statement paragraphs found")

    published_at, precision = parse_release_time(date.get_text(strip=True), release.get_text())
    return ParsedStatement(
        title=title.get_text(strip=True),
        published_at=published_at,
        published_at_precision=precision,
        paragraphs=paragraphs,
    )
