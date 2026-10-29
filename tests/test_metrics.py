import pytest

from cbce.eval.metrics import evaluate
from cbce.store.schema import Stance

H, N, D = Stance.HAWKISH, Stance.NEUTRAL, Stance.DOVISH


@pytest.mark.parametrize(
    ("truth", "predicted", "expected_macro_f1"),
    [
        pytest.param([H, N, D], [H, N, D], 1.0, id="perfect"),
        pytest.param([H, N, D], [N, D, H], 0.0, id="everything_wrong"),
        # H: P 1, R 1/2, F1 2/3. D: P 1/2, R 1, F1 2/3. N: F1 1. Macro = (2/3 + 2/3 + 1) / 3.
        pytest.param([H, H, D, N], [H, D, D, N], 7 / 9, id="worked_example"),
        # Always hawkish on [H, H, D, N]: H F1 2/3; D and N undefined so 0. Macro = 2/9.
        pytest.param([H, H, D, N], [H, H, H, H], 2 / 9, id="single_class_prediction"),
    ],
)
def test_macro_f1(truth: list[Stance], predicted: list[Stance], expected_macro_f1: float) -> None:
    # Arrange: labels worked out by hand in the case ids

    # Act
    report = evaluate(truth, predicted)

    # Assert
    assert report.macro_f1 == pytest.approx(expected_macro_f1)


def test_per_class_precision_recall_f1_and_support() -> None:
    # Arrange
    truth, predicted = [H, H, D, N], [H, D, D, N]

    # Act
    report = evaluate(truth, predicted)

    # Assert
    hawkish, dovish = report.per_class[H], report.per_class[D]
    assert (hawkish.precision, hawkish.recall, hawkish.support) == (1.0, 0.5, 2)
    assert hawkish.f1 == pytest.approx(2 / 3)
    assert (dovish.precision, dovish.recall, dovish.support) == (0.5, 1.0, 1)


def test_confusion_counts_true_against_predicted() -> None:
    # Arrange
    truth, predicted = [H, H, D, N], [H, D, D, N]

    # Act
    report = evaluate(truth, predicted)

    # Assert
    assert report.confusion == {(H, H): 1, (H, D): 1, (D, D): 1, (N, N): 1}
    assert report.n == 4


def test_a_class_with_no_true_examples_scores_zero_not_an_error() -> None:
    # Arrange: no neutral examples and no neutral predictions
    truth, predicted = [H, D], [H, D]

    # Act
    report = evaluate(truth, predicted)

    # Assert
    assert report.per_class[N].f1 == 0.0
    assert report.per_class[N].support == 0
    assert report.macro_f1 == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("truth", "predicted"),
    [
        pytest.param([H, D], [H], id="length_mismatch"),
        pytest.param([], [], id="empty"),
    ],
)
def test_invalid_input_raises(truth: list[Stance], predicted: list[Stance]) -> None:
    # Arrange: inputs that cannot be scored

    # Act / Assert
    with pytest.raises(ValueError):
        evaluate(truth, predicted)
