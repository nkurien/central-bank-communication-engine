"""Cluster bootstrap for macro F1, resampling whole groups (calendar years) with replacement.

Sentences from one year, and one meeting, are not independent, so resampling sentences would
give intervals that are too narrow. Resampling years respects that, at the price of honest,
wide intervals when there are few years. The same resamples are used for every scorer, which
makes the difference between two scorers a paired comparison.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cbce.eval.metrics import macro_f1_from_confusion
from cbce.store.schema import Stance

BOOTSTRAP_SEED = 20261029  # fixed so every report is reproducible

Confusion = Counter[tuple[Stance, Stance]]


@dataclass(frozen=True)
class Interval:
    estimate: float  # on the full sample
    low: float
    high: float


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted list, q in [0, 1]."""
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _interval(estimate: float, samples: list[float], level: float) -> Interval:
    ordered = sorted(samples)
    tail = (1 - level) / 2
    return Interval(estimate, _percentile(ordered, tail), _percentile(ordered, 1 - tail))


def bootstrap_macro_f1(
    truth: Sequence[Stance],
    predicted: Mapping[str, Sequence[Stance]],
    groups: Sequence[int],
    *,
    n_resamples: int = 2000,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> tuple[dict[str, Interval], dict[tuple[str, str], Interval]]:
    """Percentile intervals for each scorer's macro F1 and for each pair's difference.

    Returns (per_scorer, differences). A difference is keyed (a, b) with a < b by name and
    is a minus b, so an interval above zero means a beat b.
    """
    if not predicted:
        raise ValueError("no scorers to compare")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not truth or len(set(groups)) < 1:
        raise ValueError("nothing to resample")
    for name, labels in predicted.items():
        if not (len(labels) == len(truth) == len(groups)):
            raise ValueError(f"{name}: truth, predictions and groups differ in length")

    # Confusion counts per group per scorer, so each resample is a sum over a few groups.
    counts: dict[str, dict[int, Confusion]] = {name: defaultdict(Counter) for name in predicted}
    for name, labels in predicted.items():
        for true, pred, group in zip(truth, labels, groups, strict=True):
            counts[name][group][(true, pred)] += 1
    group_ids = sorted({*groups})

    def macro_f1(name: str, chosen: Sequence[int]) -> float:
        total: Confusion = Counter()
        for group in chosen:
            total.update(counts[name][group])
        return macro_f1_from_confusion(total)

    names = sorted(predicted)
    estimates = {name: macro_f1(name, group_ids) for name in names}
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(n_resamples):
        chosen = rng.choices(group_ids, k=len(group_ids))
        for name in names:
            draws[name].append(macro_f1(name, chosen))

    per_scorer = {name: _interval(estimates[name], draws[name], level) for name in names}
    differences = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            diffs = [x - y for x, y in zip(draws[a], draws[b], strict=True)]
            differences[(a, b)] = _interval(estimates[a] - estimates[b], diffs, level)
    return per_scorer, differences
