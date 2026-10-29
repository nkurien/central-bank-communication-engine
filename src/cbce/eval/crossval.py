"""Cross-validation grouped by calendar year, for scorers that are trained on dev.

Whole years are held out together, so a model never trains on a year it is scored on, and
the out-of-fold predictions cover every sentence once. The fold rule is fixed, not tuned:
years are ranked and the rank mod k is the fold.
"""

from collections.abc import Callable, Iterable, Sequence

from cbce.eval.labelled import LabelledSentence
from cbce.scoring.base import ScoreInput, Scorer
from cbce.store.schema import StanceScore

Fit = Callable[[Sequence[LabelledSentence]], Scorer]


def assign_folds(years: Iterable[int], k: int = 5) -> dict[int, int]:
    """Year -> fold number in [0, k). Neighbouring years land in different folds."""
    distinct = sorted(set(years))
    if k < 2:
        raise ValueError("need at least two folds")
    if len(distinct) < k:
        raise ValueError(f"{len(distinct)} years cannot fill {k} folds")
    return {year: rank % k for rank, year in enumerate(distinct)}


def out_of_fold_predictions(
    sentences: Sequence[LabelledSentence], fit: Fit, run_id: str, k: int = 5
) -> list[StanceScore]:
    """One prediction per sentence, each from a model that never saw that sentence's year.

    `fit` trains on the sentences it is given and returns a scorer. Results come back in
    the order of `sentences`.
    """
    fold_of_year = assign_folds((s.year for s in sentences), k)
    predictions: dict[str, StanceScore] = {}
    for fold in range(k):
        held_out = [s for s in sentences if fold_of_year[s.year] == fold]
        train = [s for s in sentences if fold_of_year[s.year] != fold]
        if {s.year for s in held_out} & {s.year for s in train}:
            raise RuntimeError("a year appears on both sides of a fold")
        scorer = fit(train)
        scored = scorer.score([ScoreInput(s.sentence_id, s.text) for s in held_out], run_id)
        predictions.update((score.sentence_id, score) for score in scored)
    return [predictions[s.sentence_id] for s in sentences]
