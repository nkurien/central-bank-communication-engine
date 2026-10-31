from collections import Counter

import pytest

from cbce.eval.crossval import assign_folds, out_of_fold_predictions
from cbce.eval.labelled import LabelledSentence
from cbce.scoring.majority import MajorityScorer
from cbce.store.schema import Stance

H, N, D = Stance.HAWKISH, Stance.NEUTRAL, Stance.DOVISH
DEV_YEARS = [1996, 1997, 1999, 2000, 2001, 2003, 2004, 2005, 2007, 2008,
             2009, 2011, 2012, 2013, 2015, 2016, 2017, 2019, 2020, 2021]  # fmt: skip


def test_the_twenty_dev_years_split_into_five_folds_of_four() -> None:
    # Arrange: the real dev years under the frozen split

    # Act
    folds = assign_folds(DEV_YEARS, 5)

    # Assert
    assert Counter(folds.values()) == {0: 4, 1: 4, 2: 4, 3: 4, 4: 4}
    assert folds[1996] == 0 and folds[1997] == 1 and folds[1999] == 2


@pytest.mark.parametrize(
    ("years", "k"),
    [
        pytest.param([2001, 2000, 2002, 2003], 2, id="input_order_does_not_matter"),
        pytest.param([2000, 2000, 2001, 2001, 2002], 3, id="duplicate_years_count_once"),
    ],
)
def test_fold_assignment_depends_only_on_the_ranked_distinct_years(
    years: list[int], k: int
) -> None:
    # Arrange / Act
    folds = assign_folds(years, k)

    # Assert
    assert folds == {y: rank % k for rank, y in enumerate(sorted(set(years)))}


@pytest.mark.parametrize(
    ("years", "k"),
    [
        pytest.param([2000, 2001], 5, id="too_few_years"),
        pytest.param([2000, 2001], 1, id="one_fold"),
    ],
)
def test_fold_assignment_rejects_impossible_requests(years: list[int], k: int) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        assign_folds(years, k)


def sentences() -> list[LabelledSentence]:
    stances = [H, N, D]
    return [LabelledSentence(f"s{i}", f"text {i}", 2000 + i % 6, stances[i % 3]) for i in range(24)]


def test_every_sentence_is_predicted_once_in_input_order() -> None:
    # Arrange
    data = sentences()

    # Act
    predictions = out_of_fold_predictions(
        data, lambda train: MajorityScorer.fit(s.stance for s in train), "run-1", k=3
    )

    # Assert
    assert [p.sentence_id for p in predictions] == [s.sentence_id for s in data]
    assert all(p.run_id == "run-1" for p in predictions)


def test_a_model_never_trains_on_the_year_it_is_scored_on() -> None:
    # Arrange: record the years each fit() call is given
    seen_in_training: list[set[int]] = []

    def fit(train):
        seen_in_training.append({s.year for s in train})
        return MajorityScorer.fit(s.stance for s in train)

    data = sentences()

    # Act
    out_of_fold_predictions(data, fit, "r", k=3)

    # Assert: each call trained on 4 of the 6 years, and across calls every year was left out
    assert len(seen_in_training) == 3
    assert all(len(years) == 4 for years in seen_in_training)
    assert set.union(*({2000 + i for i in range(6)} - years for years in seen_in_training)) == {
        2000 + i for i in range(6)
    }


def test_held_out_predictions_come_from_a_model_fitted_without_those_sentences() -> None:
    # Arrange: years 2000-2005; make year 2000's sentences all hawkish and the rest all dovish
    data = [LabelledSentence(f"s{i}", "t", 2000 + i % 6, H if i % 6 == 0 else D) for i in range(24)]

    # Act
    predictions = out_of_fold_predictions(
        data, lambda train: MajorityScorer.fit(s.stance for s in train), "r", k=6
    )

    # Assert: a model that never saw 2000 cannot predict hawkish for it
    by_id = {p.sentence_id: p.stance for p in predictions}
    assert all(by_id[f"s{i}"] is D for i in range(0, 24, 6))


def test_repeated_sentence_ids_are_refused() -> None:
    # Arrange
    data = sentences()
    data[1] = LabelledSentence(data[0].sentence_id, "other text", 2001, N)

    # Act / Assert
    with pytest.raises(ValueError, match="not unique"):
        out_of_fold_predictions(
            data, lambda train: MajorityScorer.fit(s.stance for s in train), "run-1", k=3
        )
