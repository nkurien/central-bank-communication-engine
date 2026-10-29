"""Parse a saved Federal Reserve FOMC statement page."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup, Tag

NEW_YORK = ZoneInfo("America/New_York")

# Paragraphs that follow the policy text and carry no stance.
_BOILERPLATE = re.compile(
    r"^(Voting for|Voting against|For media inquiries|Implementation Note|In a related action"
    r"|In taking the discount rate action)",
    re.IGNORECASE,
)
_RELEASE_TIME = re.compile(r"(\d{1,2}):(\d{2})\s*([ap])\.m\.", re.IGNORECASE)
_LEGACY_DATE = re.compile(r"Release Date:\s*([A-Z][a-z]+ \d{1,2}, \d{4})")
_TITLE_SEPARATOR = re.compile(r"\s*--\s*")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_LEGACY_PARAGRAPH = re.compile(r"<p\b[^>]*>", re.IGNORECASE)
_LEGACY_TAIL = re.compile(r"<hr\b|<a\s+href=[\"']\.\./", re.IGNORECASE)
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


def _text(node: Tag | BeautifulSoup) -> str:
    """Whitespace-normalised text, or empty if the node is nothing but link text."""
    text = " ".join(node.get_text().split())
    link_text = " ".join(" ".join(a.get_text() for a in node.find_all("a")).split())
    return "" if text == link_text else text


def _clean_paragraphs(nodes: Iterable[Tag | BeautifulSoup]) -> tuple[str, ...]:
    texts = (_text(node) for node in nodes)
    return tuple(text for text in texts if text and not _BOILERPLATE.match(text))


def _parse_current(article: Tag) -> ParsedStatement:
    """Layout used from the late 2000s: an #article block with classed date, title and body."""
    date = article.find(class_="article__time")
    title = article.find(class_="title")
    release = article.find(class_="releaseTime")
    if not (date and title and release):
        raise ParseError("missing date, title or release time")

    # The body is the text column. Skip the heading column and attachment panels, which
    # share the column class but hold no statement text.
    columns = [
        div
        for div in article.find_all("div", class_="col-md-8")
        if not {"heading", "panel"} & set(div.get("class", []))
    ]
    if not columns:
        raise ParseError("no body column found")
    body = max(columns, key=lambda div: len(div.find_all("p")))
    published_at, precision = parse_release_time(date.get_text(strip=True), release.get_text())
    return ParsedStatement(
        title=title.get_text(strip=True),
        published_at=published_at,
        published_at_precision=precision,
        paragraphs=_clean_paragraphs(body.find_all("p")),
    )


def _parse_legacy(html: str, soup: BeautifulSoup) -> ParsedStatement:
    """Layout used until about 2005: a plain table with unclosed <p> tags."""
    # "FRB: Press Release -- FOMC statement -- January 29, 2003"; the spaces around the
    # dashes are missing on some pages.
    parts = _TITLE_SEPARATOR.split(soup.title.get_text()) if soup.title else []
    if len(parts) < 3:
        raise ParseError("unrecognised page title")
    title = " -- ".join(parts[1:-1]).strip()

    # The body is the table cell with the most paragraphs. Everything above it, including
    # a separate small table holding "For immediate release", is the page head.
    cells = list(re.finditer(r"<td[^>]*>(.*?)</td>", html, re.IGNORECASE | re.DOTALL))
    if not cells:
        raise ParseError("no body cell")
    cell = max(cells, key=lambda m: len(_LEGACY_PARAGRAPH.findall(m[1])))
    head = html[: cell.start()]
    date = _LEGACY_DATE.search(head)
    if date is None:
        raise ParseError("no release date")
    published_at, precision = parse_release_time(date[1], head[date.end() :])

    body = _COMMENT.sub("", cell[1])
    tail = _LEGACY_TAIL.search(body)  # navigation after the last paragraph
    if tail:
        body = body[: tail.start()]
    chunks = _LEGACY_PARAGRAPH.split(body)
    return ParsedStatement(
        title=title,
        published_at=published_at,
        published_at_precision=precision,
        paragraphs=_clean_paragraphs(BeautifulSoup(chunk, "html.parser") for chunk in chunks),
    )


def parse_statement(html: str) -> ParsedStatement:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find(id="article")
    parsed = _parse_current(article) if article is not None else _parse_legacy(html, soup)
    if not parsed.paragraphs:
        raise ParseError("no statement paragraphs found")
    return parsed
