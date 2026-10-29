"""Record types for the store tables. One pydantic model per table."""

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Bank(StrEnum):
    FED = "fed"
    BOE = "boe"
    ECB = "ecb"


class DocType(StrEnum):
    STATEMENT = "statement"
    MINUTES = "minutes"
    PRESS_CONFERENCE = "press_conference"
    SPEECH = "speech"
    OTHER_RELEASE = "other_release"  # liquidity, facility or strategy announcements


class Stance(StrEnum):
    HAWKISH = "hawkish"
    NEUTRAL = "neutral"
    DOVISH = "dovish"


class Topic(StrEnum):
    INFLATION = "inflation"
    LABOUR = "labour"
    GROWTH = "growth"
    FINANCIAL_STABILITY = "financial_stability"
    FORWARD_GUIDANCE = "forward_guidance"
    BALANCE_SHEET = "balance_sheet"
    OTHER = "other"


class Record(BaseModel):
    """Base for all table records: immutable, no unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class Document(Record):
    """One released text.

    `published_at` is the release timestamp from the source, never the fetch time.
    `event_ts` is when the market could first have reacted. It equals `published_at` for
    statements and minutes, and precedes it for press conferences, whose transcripts are
    posted after the event. Older pages give only a date; `published_at_precision` is "day"
    and `published_at` is the end of that day, so it is never earlier than the real release.
    """

    doc_id: str  # hash of the parsed text, stable across fetches
    bank: Bank
    doc_type: DocType
    published_at: datetime
    published_at_precision: Literal["minute", "day"] = "minute"
    event_ts: datetime
    source_url: str
    raw_path: str
    fetched_at: datetime

    _aware = field_validator("published_at", "event_ts", "fetched_at")(_require_aware)

    @model_validator(mode="after")
    def _point_in_time(self) -> Self:
        if self.event_ts > self.published_at:
            raise ValueError("event_ts cannot be after published_at")
        if self.published_at > self.fetched_at:
            raise ValueError("published_at cannot be after fetched_at")
        return self


class Sentence(Record):
    sentence_id: str
    doc_id: str
    position: int
    text: str
    section: str | None = None


class LLMCall(Record):
    call_id: str  # hash of (model, model_version, prompt_version, input hash)
    model: str
    model_version: str
    prompt_version: str
    raw_response: str
    tokens: int
    created_at: datetime

    _aware = field_validator("created_at")(_require_aware)


class StanceScore(Record):
    sentence_id: str
    scorer: str  # llm, dictionary, finbert, encoder, majority
    topic: Topic
    stance: Stance
    confidence: float
    evidence: str | None = None
    run_id: str

    @field_validator("confidence")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return value


class DocScore(Record):
    doc_id: str
    scorer: str
    topic: Topic | None = None  # None means the overall score
    score: float  # share hawkish minus share dovish, confidence-weighted
    diff_vs_previous: float | None = None  # None for the first release of a type
    run_id: str


class YieldObs(Record):
    series: str
    date: date
    value: float
    fetched_at: datetime

    _aware = field_validator("fetched_at")(_require_aware)


class Event(Record):
    event_id: str
    bank: Bank
    doc_id: str
    release_ts: datetime
    window_start: date
    window_end: date
    yield_change_2y: float | None = None
    yield_change_10y: float | None = None

    _aware = field_validator("release_ts")(_require_aware)

    @model_validator(mode="after")
    def _window_order(self) -> Self:
        if self.window_start >= self.window_end:
            raise ValueError("window_start must be before window_end")
        return self


class Run(Record):
    run_id: str
    git_sha: str
    config_hash: str
    started_at: datetime
    metrics_json: str

    _aware = field_validator("started_at")(_require_aware)
