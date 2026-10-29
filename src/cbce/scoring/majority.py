"""Majority-class baseline: always predicts the most common label. The floor for accuracy."""

from collections import Counter
from collections.abc import Iterable, Sequence

from cbce.scoring.base import ScoreInput
from cbce.store.schema import Stance, StanceScore, Topic


class MajorityScorer:
    name = "majority"

    def __init__(self, stance: Stance, share: float) -> None:
        self.stance = stance
        self.share = share  # fraction of the fitting data with this label

    @classmethod
    def fit(cls, stances: Iterable[Stance]) -> "MajorityScorer":
        counts = Counter(stances)
        if not counts:
            raise ValueError("cannot fit on no labels")
        # Ties break alphabetically on the label so the result never depends on input order.
        stance, count = min(counts.items(), key=lambda item: (-item[1], item[0].value))
        return cls(stance, count / sum(counts.values()))

    def score(self, items: Sequence[ScoreInput], run_id: str) -> list[StanceScore]:
        return [
            StanceScore(
                sentence_id=item.sentence_id,
                scorer=self.name,
                topic=Topic.OTHER,
                stance=self.stance,
                confidence=self.share,
                run_id=run_id,
            )
            for item in items
        ]
