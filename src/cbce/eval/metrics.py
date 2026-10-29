"""Sentence-level metrics: macro F1, per-class precision, recall and F1, confusion matrix."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cbce.store.schema import Stance

LABELS = (Stance.HAWKISH, Stance.NEUTRAL, Stance.DOVISH)


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int  # true examples of this class


@dataclass(frozen=True)
class Report:
    n: int
    macro_f1: float
    per_class: dict[Stance, ClassMetrics]
    confusion: dict[tuple[Stance, Stance], int]  # (true, predicted) -> count


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0  # undefined scores count as 0


def macro_f1_from_confusion(confusion: Mapping[tuple[Stance, Stance], int]) -> float:
    """Macro F1 from (true, predicted) counts. A class with no predictions or no examples
    scores 0, the same convention as `evaluate`.
    """
    total = 0.0
    for label in LABELS:
        true_positive = confusion.get((label, label), 0)
        predicted_total = sum(confusion.get((t, label), 0) for t in LABELS)
        support = sum(confusion.get((label, p), 0) for p in LABELS)
        total += _ratio(2 * true_positive, predicted_total + support)
    return total / len(LABELS)


def evaluate(truth: Sequence[Stance], predicted: Sequence[Stance]) -> Report:
    if len(truth) != len(predicted):
        raise ValueError(f"{len(truth)} labels but {len(predicted)} predictions")
    if not truth:
        raise ValueError("nothing to evaluate")

    confusion = Counter(zip(truth, predicted, strict=True))
    per_class: dict[Stance, ClassMetrics] = {}
    for label in LABELS:
        true_positive = confusion[(label, label)]
        predicted_total = sum(confusion[(t, label)] for t in LABELS)
        support = sum(confusion[(label, p)] for p in LABELS)
        precision = _ratio(true_positive, predicted_total)
        recall = _ratio(true_positive, support)
        f1 = _ratio(2 * true_positive, predicted_total + support)  # = 2PR / (P + R)
        per_class[label] = ClassMetrics(precision, recall, f1, support)

    macro_f1 = sum(m.f1 for m in per_class.values()) / len(LABELS)
    return Report(len(truth), macro_f1, per_class, dict(confusion))
