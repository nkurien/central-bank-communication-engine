import pytest
from huggingface_hub import try_to_load_from_cache

from cbce.scoring.base import ScoreInput
from cbce.scoring.finbert import FINBERT_MODEL, FINBERT_REVISION, FinBertScorer
from cbce.store.schema import Stance, Topic

H, N, D = Stance.HAWKISH, Stance.NEUTRAL, Stance.DOVISH


def fixed(*dicts: dict[str, float]):
    """A fake predictor returning the given probability dicts in order."""

    def predict(texts):
        assert len(texts) == len(dicts)
        return list(dicts)

    return predict


@pytest.mark.parametrize(
    ("probabilities", "stance", "confidence"),
    [
        pytest.param({"positive": 0.7, "negative": 0.2, "neutral": 0.1}, H, 0.7, id="positive"),
        pytest.param({"positive": 0.1, "negative": 0.8, "neutral": 0.1}, D, 0.8, id="negative"),
        pytest.param({"positive": 0.2, "negative": 0.1, "neutral": 0.7}, N, 0.7, id="neutral"),
        pytest.param({"positive": 0.4, "negative": 0.4, "neutral": 0.2}, H, 0.4, id="tie_by_name"),
    ],
)
def test_sentiment_label_is_read_as_stance(
    probabilities: dict[str, float], stance: Stance, confidence: float
) -> None:
    # Arrange
    scorer = FinBertScorer(fixed(probabilities))

    # Act
    (score,) = scorer.score([ScoreInput("s1", "text")], "run-1")

    # Assert
    assert score.stance is stance
    assert score.confidence == pytest.approx(confidence)
    assert (score.scorer, score.topic, score.run_id) == ("finbert", Topic.OTHER, "run-1")


def test_unexpected_label_raises() -> None:
    # Arrange: a model whose labels are not the three expected sentiments
    scorer = FinBertScorer(fixed({"bullish": 0.9, "bearish": 0.1}))

    # Act / Assert
    with pytest.raises(ValueError, match="unexpected FinBERT label"):
        scorer.score([ScoreInput("s", "text")], "r")


def test_empty_input_never_calls_the_model() -> None:
    # Arrange
    scorer = FinBertScorer(lambda texts: pytest.fail("model should not be called"))

    # Act
    result = scorer.score([], "r")

    # Assert
    assert result == []


def test_results_keep_input_order() -> None:
    # Arrange
    scorer = FinBertScorer(
        fixed(
            {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
            {"positive": 0.05, "negative": 0.9, "neutral": 0.05},
        )
    )
    items = [ScoreInput("first", "a"), ScoreInput("second", "b")]

    # Act
    scores = scorer.score(items, "r")

    # Assert
    assert [(s.sentence_id, s.stance) for s in scores] == [("first", H), ("second", D)]


model_cached = try_to_load_from_cache(FINBERT_MODEL, "model.safetensors", revision=FINBERT_REVISION)


@pytest.mark.skipif(not isinstance(model_cached, str), reason="pinned FinBERT not in local cache")
def test_pinned_model_loads_offline_and_scores_deterministically() -> None:
    # Arrange: the real model, loaded with local_files_only so this never touches the network
    scorer = FinBertScorer.load(local_files_only=True, batch_size=2)
    items = [
        ScoreInput(f"s{i}", text)
        for i, text in enumerate(
            [
                "Inflation has risen and the Committee will tighten policy.",
                "Growth was weak.",
                "Met.",
            ]
        )
    ]

    # Act
    first = scorer.score(items, "r")
    second = scorer.score(items, "r")

    # Assert
    assert [s.sentence_id for s in first] == ["s0", "s1", "s2"]
    assert all(0.0 <= s.confidence <= 1.0 for s in first)
    assert [(s.stance, s.confidence) for s in first] == [(s.stance, s.confidence) for s in second]
