"""LLM stance scorer: cached, schema-validated, one retry (hard rules 4, 5 and 6).

Every attempt is cached in `llm_calls` under (model, model_version, prompt_version, input hash,
attempt), so re-running never scores the same input twice, failed attempts included. A response
that fails the schema gets one retry; if that fails too the sentence is logged in `excluded`
and gets no score. Model output is never patched by hand.
"""

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cbce.scoring.base import ScoreInput
from cbce.scoring.llm_client import LLMClient, LLMResponse
from cbce.store.schema import Stance, StanceScore, Topic

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # the first call and one retry


class LLMStance(BaseModel):
    """What the model must return for one sentence."""

    model_config = ConfigDict(extra="forbid")

    topic: Topic
    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str | None = None


@dataclass(frozen=True)
class Prompt:
    """A frozen prompt file. The version is the file stem, so a change means a new file."""

    version: str
    system: str
    user_template: str  # contains {sentence}

    @classmethod
    def load(cls, path: Path) -> "Prompt":
        system, user = path.read_text().split("\n---\n", maxsplit=1)
        return cls(version=path.stem, system=system.strip(), user_template=user.strip())

    def render(self, sentence: str) -> str:
        # replace, not format: the template may contain literal braces
        return self.user_template.replace("{sentence}", sentence)


class CallCache:
    """Insert-only view of `llm_calls`. A stored response is never edited or deleted."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._connection = connection

    def get(self, call_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT raw_response FROM llm_calls WHERE call_id = ?", [call_id]
        ).fetchone()
        return None if row is None else row[0]

    def put(
        self,
        call_id: str,
        client: LLMClient,
        prompt_version: str,
        response: LLMResponse,
        now: datetime,
    ) -> None:
        self._connection.execute(
            "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                call_id,
                client.model,
                client.model_version,
                prompt_version,
                response.text,
                response.tokens,
                now,
            ],
        )


def call_id(client: LLMClient, prompt_version: str, text: str, attempt: int) -> str:
    input_hash = hashlib.sha256(text.encode()).hexdigest()
    key = "\x1f".join(
        [client.model, client.model_version, prompt_version, input_hash, str(attempt)]
    )
    return hashlib.sha256(key.encode()).hexdigest()


class LLMScorer:
    name = "llm"

    def __init__(
        self,
        client: LLMClient,
        prompt: Prompt,
        cache: CallCache,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self._prompt = prompt
        self._cache = cache
        self._clock = clock
        self._schema = LLMStance.model_json_schema()
        self.excluded: list[str] = []  # sentence_ids that failed validation twice

    def score(self, items: Sequence[ScoreInput], run_id: str) -> list[StanceScore]:
        """Scores in input order. Excluded sentences are left out and listed in `excluded`."""
        scores = []
        for item in items:
            parsed = self._score_one(item)
            if parsed is None:
                self.excluded.append(item.sentence_id)
                log.warning(
                    "excluded %s: no valid response after %d attempts",
                    item.sentence_id,
                    MAX_ATTEMPTS,
                )
                continue
            scores.append(
                StanceScore(
                    sentence_id=item.sentence_id,
                    scorer=self.name,
                    topic=parsed.topic,
                    stance=parsed.stance,
                    confidence=parsed.confidence,
                    evidence=parsed.evidence,
                    run_id=run_id,
                )
            )
        return scores

    def _score_one(self, item: ScoreInput) -> LLMStance | None:
        for attempt in range(MAX_ATTEMPTS):
            key = call_id(self._client, self._prompt.version, item.text, attempt)
            raw = self._cache.get(key)
            if raw is None:
                response = self._client.complete(
                    self._prompt.system, self._prompt.render(item.text), self._schema
                )
                self._cache.put(key, self._client, self._prompt.version, response, self._clock())
                raw = response.text
            try:
                return LLMStance.model_validate_json(raw)
            except ValidationError:
                continue
        return None
