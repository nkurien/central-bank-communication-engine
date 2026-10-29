import pytest

from cbce.sources.base import FetchResult
from cbce.sources.fetch import CONTACT_ENV, rate_limited, user_agent


class FakeTime:
    """Deterministic clock whose sleep advances time and records the waits."""

    def __init__(self) -> None:
        self.now = 100.0
        self.waits: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


def echo(url: str) -> FetchResult:
    return FetchResult(url.encode(), url)


@pytest.mark.parametrize(
    ("gap_between_calls", "expected_waits"),
    [
        pytest.param(0.0, [2.0, 2.0], id="back_to_back"),
        pytest.param(0.5, [1.5, 1.5], id="partial_gap"),
        pytest.param(2.0, [], id="gap_equals_interval"),
        pytest.param(5.0, [], id="gap_exceeds_interval"),
    ],
)
def test_rate_limiter_spaces_calls(gap_between_calls: float, expected_waits: list[float]) -> None:
    # Arrange
    time = FakeTime()
    fetch = rate_limited(echo, 2.0, clock=time.clock, sleep=time.sleep)

    # Act
    for url in ("a", "b", "c"):
        fetch(url)
        time.now += gap_between_calls

    # Assert
    assert time.waits == expected_waits


def test_rate_limiter_does_not_delay_the_first_call() -> None:
    # Arrange
    time = FakeTime()
    fetch = rate_limited(echo, 10.0, clock=time.clock, sleep=time.sleep)

    # Act
    result = fetch("first")

    # Assert
    assert result.content == b"first"
    assert time.waits == []


def test_user_agent_requires_a_contact(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv(CONTACT_ENV, raising=False)

    # Act / Assert
    with pytest.raises(RuntimeError, match=CONTACT_ENV):
        user_agent()


@pytest.mark.parametrize("contact", ["me@example.test", "https://example.test/about"])
def test_user_agent_names_the_project_and_contact(
    monkeypatch: pytest.MonkeyPatch, contact: str
) -> None:
    # Arrange
    monkeypatch.setenv(CONTACT_ENV, contact)

    # Act
    agent = user_agent()

    # Assert
    assert agent.startswith("cbce-research/")
    assert contact in agent
