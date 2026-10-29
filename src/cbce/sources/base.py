"""Shared interface every bank adapter implements."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Protocol

from cbce.store.schema import Bank, DocType


@dataclass(frozen=True)
class Release:
    """A release found by discovery, before it has been fetched."""

    bank: Bank
    doc_type: DocType
    url: str
    release_date: date


@dataclass(frozen=True)
class FetchResult:
    content: bytes
    final_url: str  # after redirects


# A fetcher takes a URL and returns bytes. Tests pass fakes, so no test touches the network.
Fetcher = Callable[[str], FetchResult]


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    published_at: datetime
    published_at_precision: Literal["minute", "day"]
    event_ts: datetime
    paragraphs: tuple[str, ...]


class SourceAdapter(Protocol):
    bank: Bank

    def discover(self, fetch: Fetcher) -> list[Release]:
        """List releases, oldest first. Idempotent."""
        ...

    def parse(self, release: Release, content: bytes) -> ParsedDocument:
        """Turn fetched bytes into text and timestamps. Raises on unexpected structure."""
        ...
