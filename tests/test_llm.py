import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from cbce.scoring.base import ScoreInput
from cbce.scoring.llm import CallCache, LLMScorer, Prompt, call_id
from cbce.scoring.llm_client import LLMResponse
from cbce.store.ddl import create_schema
from cbce.store.schema import Stance, Topic

NOW = datetime(2026, 10, 30, tzinfo=UTC)
GOOD = json.dumps(
    {"topic": "inflation", "stance": "hawkish", "confidence": 0.8, "evidence": "too high"}
)
PROMPT = Prompt(version="stance_v1", system="sys", user_template="Sentence: {sentence}")


class FakeClient:
    model = "fake"
    model_version = "digest-1"

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.calls = 0

    def complete(self, system, user, schema):
        self.calls += 1
        return LLMResponse(text=self._texts.pop(0), tokens=10)


def make_scorer(client: FakeClient) -> tuple[LLMScorer, duckdb.DuckDBPyConnection]:
    connection = duckdb.connect(":memory:")
    create_schema(connection)
    return LLMScorer(client, PROMPT, CallCache(connection), clock=lambda: NOW), connection


def test_valid_response_becomes_a_stance_score() -> None:
    # Arrange
    scorer, _ = make_scorer(FakeClient(GOOD))

    # Act
    (score,) = scorer.score([ScoreInput("s1", "Inflation is too high.")], "run-1")

    # Assert
    assert (score.topic, score.stance, score.confidence) == (Topic.INFLATION, Stance.HAWKISH, 0.8)
    assert (score.scorer, score.run_id, score.evidence) == ("llm", "run-1", "too high")


@pytest.mark.parametrize(
    ("responses", "calls", "scored"),
    [
        pytest.param(["not json", GOOD], 2, True, id="retry_succeeds"),
        pytest.param(['{"topic": "inflation"}', GOOD], 2, True, id="missing_fields_then_ok"),
        pytest.param(
            [GOOD.replace('"hawkish"', '"bullish"'), GOOD.replace("0.8", "1.7")],
            2,
            False,
            id="both_invalid_excluded",
        ),
    ],
)
def test_invalid_response_is_retried_once_then_excluded(
    responses: list[str], calls: int, scored: bool
) -> None:
    # Arrange
    client = FakeClient(*responses)
    scorer, _ = make_scorer(client)

    # Act
    scores = scorer.score([ScoreInput("s1", "text")], "run-1")

    # Assert
    assert client.calls == calls
    assert bool(scores) is scored
    assert scorer.excluded == ([] if scored else ["s1"])


def test_rescoring_the_same_input_makes_no_new_calls() -> None:
    # Arrange
    client = FakeClient("not json", "still not json")
    scorer, connection = make_scorer(client)
    items = [ScoreInput("s1", "text")]
    scorer.score(items, "run-1")

    # Act
    again = LLMScorer(client, PROMPT, CallCache(connection), clock=lambda: NOW)
    again.score(items, "run-2")

    # Assert
    assert client.calls == 2  # failed attempts are cached too
    assert again.excluded == ["s1"]


@pytest.mark.parametrize(
    ("changed", "other"),
    [
        pytest.param(("model_version", "digest-2"), None, id="model_version"),
        pytest.param(("prompt_version", "stance_v2"), None, id="prompt_version"),
        pytest.param(("text", "other text"), None, id="input"),
        pytest.param(("attempt", 1), None, id="attempt"),
    ],
)
def test_call_id_depends_on_every_key_part(changed: tuple[str, object], other: None) -> None:
    # Arrange
    base = {
        "model_version": "digest-1",
        "prompt_version": "stance_v1",
        "text": "text",
        "attempt": 0,
    }

    def make(parts: dict) -> str:
        client = FakeClient()
        client.model_version = parts["model_version"]
        return call_id(client, parts["prompt_version"], parts["text"], parts["attempt"])

    # Act
    altered = make({**base, changed[0]: changed[1]})

    # Assert
    assert altered != make(base)


def test_responses_are_stored_with_their_key_parts() -> None:
    # Arrange
    scorer, connection = make_scorer(FakeClient(GOOD))

    # Act
    scorer.score([ScoreInput("s1", "text")], "run-1")

    # Assert
    row = connection.execute(
        "SELECT model, model_version, prompt_version, raw_response, tokens FROM llm_calls"
    ).fetchone()
    assert row == ("fake", "digest-1", "stance_v1", GOOD, 10)


def test_prompt_file_is_split_into_system_and_user(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "stance_v9.md"
    path.write_text('Be brief. {"json": true}\n---\nSentence: {sentence}\n')

    # Act
    prompt = Prompt.load(path)

    # Assert
    assert prompt.version == "stance_v9"
    assert prompt.system == 'Be brief. {"json": true}'
    assert prompt.render("Rates rose.") == "Sentence: Rates rose."


def test_shipped_prompt_loads() -> None:
    # Arrange
    path = Path(__file__).parent.parent / "prompts" / "stance_v1.md"

    # Act
    prompt = Prompt.load(path)

    # Assert
    assert "{sentence}" in prompt.user_template
