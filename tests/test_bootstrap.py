import pytest

from cbce.eval.bootstrap import Interval, _percentile, bootstrap_macro_f1
from cbce.eval.metrics import evaluate, macro_f1_from_confusion
from cbce.store.schema import Stance

H, N, D = Stance.HAWKISH, Stance.NEUTRAL, Stance.DOVISH
CYCLE = [H, N, D]


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        pytest.param([1.0, 2.0, 3.0, 4.0], 0.0, 1.0, id="minimum"),
        pytest.param([1.0, 2.0, 3.0, 4.0], 1.0, 4.0, id="maximum"),
        pytest.param([1.0, 2.0, 3.0, 4.0], 0.5, 2.5, id="median_interpolates"),
        pytest.param([5.0], 0.975, 5.0, id="single_value"),
    ],
)
def test_percentile(values: list[float], q: float, expected: float) -> None:
    # Arrange: a sorted list

    # Act
    result = _percentile(values, q)

    # Assert
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("confusion", "expected"),
    [
        pytest.param({(H, H): 1, (H, D): 1, (D, D): 1, (N, N): 1}, 7 / 9, id="worked_example"),
        pytest.param({(H, H): 2}, 1 / 3, id="one_class_only_others_score_zero"),
        pytest.param({}, 0.0, id="empty"),
    ],
)
def test_macro_f1_from_confusion(confusion: dict, expected: float) -> None:
    # Arrange: (true, predicted) counts

    # Act
    result = macro_f1_from_confusion(confusion)

    # Assert
    assert result == pytest.approx(expected)


def make_data(n_groups: int = 6, per_group: int = 9):
    truth = [CYCLE[i % 3] for i in range(n_groups * per_group)]
    groups = [i // per_group for i in range(len(truth))]
    return truth, groups


def test_point_estimate_matches_the_plain_evaluation() -> None:
    # Arrange: an imperfect scorer over several groups
    truth, groups = make_data()
    predicted = [t if i % 4 else CYCLE[(CYCLE.index(t) + 1) % 3] for i, t in enumerate(truth)]

    # Act
    per_scorer, _ = bootstrap_macro_f1(truth, {"s": predicted}, groups, n_resamples=50)

    # Assert
    assert per_scorer["s"].estimate == pytest.approx(evaluate(truth, predicted).macro_f1)


def test_a_perfect_scorer_has_a_degenerate_interval_at_one() -> None:
    # Arrange
    truth, groups = make_data()

    # Act
    per_scorer, _ = bootstrap_macro_f1(truth, {"s": list(truth)}, groups, n_resamples=100)

    # Assert
    assert per_scorer["s"] == Interval(1.0, 1.0, 1.0)


def test_identical_groups_give_a_zero_width_interval() -> None:
    # Arrange: every group has the same composition and the same errors
    truth, groups = make_data()
    predicted = [t if i % 9 else H for i, t in enumerate(truth)]  # one fixed error per group

    # Act
    per_scorer, _ = bootstrap_macro_f1(truth, {"s": predicted}, groups, n_resamples=100)

    # Assert
    interval = per_scorer["s"]
    assert interval.low == pytest.approx(interval.estimate)
    assert interval.high == pytest.approx(interval.estimate)


def test_uneven_groups_widen_the_interval_around_the_estimate() -> None:
    # Arrange: perfect in half the groups, always neutral in the other half
    truth, groups = make_data(n_groups=8)
    predicted = [t if g % 2 else N for t, g in zip(truth, groups, strict=True)]

    # Act
    per_scorer, _ = bootstrap_macro_f1(truth, {"s": predicted}, groups, n_resamples=500)

    # Assert
    interval = per_scorer["s"]
    assert interval.low < interval.estimate < interval.high
    assert interval.high - interval.low > 0.2


@pytest.mark.parametrize(
    ("better", "worse", "key", "expected_difference"),
    [
        pytest.param("alpha", "zeta", ("alpha", "zeta"), 1.0, id="first_name_wins"),
        pytest.param(
            "zeta", "alpha", ("alpha", "zeta"), -1.0, id="difference_is_first_name_minus_second"
        ),
    ],
)
def test_difference_is_first_name_minus_second(
    better: str, worse: str, key: tuple[str, str], expected_difference: float
) -> None:
    # Arrange: one scorer perfect, one always wrong
    truth, groups = make_data()
    wrong = [CYCLE[(CYCLE.index(t) + 1) % 3] for t in truth]

    # Act
    _, differences = bootstrap_macro_f1(
        truth, {better: list(truth), worse: wrong}, groups, n_resamples=50
    )

    # Assert
    assert differences[key] == Interval(
        expected_difference, expected_difference, expected_difference
    )


def graded_errors(n_groups: int = 10, per_group: int = 9):
    """Group g gets g wrong predictions, so groups differ in accuracy by small steps."""
    truth, groups = make_data(n_groups, per_group)
    predicted = []
    for i, (t, g) in enumerate(zip(truth, groups, strict=True)):
        wrong = (i % per_group) < g
        predicted.append(CYCLE[(CYCLE.index(t) + 1) % 3] if wrong else t)
    return truth, predicted, groups


def test_same_seed_reproduces_and_a_different_seed_changes_the_bounds() -> None:
    # Arrange
    truth, predicted, groups = graded_errors()

    # Act
    first, _ = bootstrap_macro_f1(truth, {"s": predicted}, groups, n_resamples=200, seed=1)
    again, _ = bootstrap_macro_f1(truth, {"s": predicted}, groups, n_resamples=200, seed=1)
    other, _ = bootstrap_macro_f1(truth, {"s": predicted}, groups, n_resamples=200, seed=2)

    # Assert
    assert first == again
    assert first["s"] != other["s"]
    assert first["s"].estimate == other["s"].estimate


def test_scorers_are_resampled_together() -> None:
    # Arrange: identical predictions must produce an identical (zero) difference every time
    truth, groups = make_data(n_groups=8)
    predicted = [t if g % 2 else N for t, g in zip(truth, groups, strict=True)]

    # Act
    _, differences = bootstrap_macro_f1(
        truth, {"a": predicted, "b": list(predicted)}, groups, n_resamples=200
    )

    # Assert
    assert differences[("a", "b")] == Interval(0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"predicted": {}}, id="no_scorers"),
        pytest.param({"level": 1.0}, id="level_one"),
        pytest.param({"level": 0.0}, id="level_zero"),
        pytest.param({"n_resamples": 0}, id="no_resamples"),
        pytest.param({"groups": [0, 1]}, id="groups_length_mismatch"),
        pytest.param({"predicted": {"s": [H]}}, id="predictions_length_mismatch"),
        pytest.param({"truth": [], "predicted": {"s": []}, "groups": []}, id="empty_data"),
    ],
)
def test_invalid_input_raises(kwargs: dict) -> None:
    # Arrange: start from valid inputs and break one thing
    truth, groups = make_data()
    arguments = {"truth": truth, "predicted": {"s": list(truth)}, "groups": groups}
    arguments.update(kwargs)

    # Act / Assert
    with pytest.raises(ValueError):
        bootstrap_macro_f1(
            arguments.pop("truth"),
            arguments.pop("predicted"),
            arguments.pop("groups"),
            **arguments,
        )
