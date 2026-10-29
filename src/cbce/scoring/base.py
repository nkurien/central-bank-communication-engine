"""The one interface every scorer implements: the LLM, the baselines and any later model."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from cbce.store.schema import StanceScore


@dataclass(frozen=True)
class ScoreInput:
    sentence_id: str
    text: str


class Scorer(Protocol):
    name: str

    def score(self, items: Sequence[ScoreInput], run_id: str) -> list[StanceScore]:
        """One record per input, in input order. Document scores are computed elsewhere."""
        ...
