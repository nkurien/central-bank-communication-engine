import pytest

from cbce.scoring.base import ScoreInput
from cbce.scoring.majority import MajorityScorer
from cbce.store.schema import Stance, Topic

H, N, D = Stance.HAWKISH, Stance.NEUTRAL, Stance.DOVISH


@pytest.mark.parametrize(
    ("labels", "expected", "share"),
    [
        pytest.param([H, H, D], H, 2 / 3, id="clear_majority"),
        pytest.param([N], N, 1.0, id="single_label"),
        pytest.param([D, D, D, H, N], D, 3 / 5, id="plurality"),
        pytest.param([H, D], D, 0.5, id="tie_breaks_alphabetically_dovish_first"),
        pytest.param([D, H], D, 0.5, id="tie_does_not_depend_on_order"),
        pytest.param([H, N, D], D, 1 / 3, id="three_way_tie"),
    ],
)
def test_fit_picks_the_most_common_label(
    labels: list[Stance], expected: Stance, share: float
) -> None:
    # Arrange: labels from the fitting data

    # Act
    scorer = MajorityScorer.fit(labels)

    # Assert
    assert scorer.stance is expected
    assert scorer.share == pytest.approx(share)


def test_fit_on_nothing_raises() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        MajorityScorer.fit([])


@pytest.mark.parametrize("count", [0, 1, 5])
def test_score_returns_one_identical_prediction_per_input_in_order(count: int) -> None:
    # Arrange
    scorer = MajorityScorer.fit([H, H, D])
    items = [ScoreInput(f"s{i}", f"text {i}") for i in range(count)]

    # Act
    scores = scorer.score(items, "run-1")

    # Assert
    assert [s.sentence_id for s in scores] == [item.sentence_id for item in items]
    assert all(s.stance is H and s.scorer == "majority" for s in scores)
    assert all(s.topic is Topic.OTHER and s.run_id == "run-1" for s in scores)
    assert all(s.confidence == pytest.approx(2 / 3) for s in scores)
