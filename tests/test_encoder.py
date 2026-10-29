import shutil
from pathlib import Path

import pytest
from huggingface_hub import try_to_load_from_cache

from cbce.scoring.base import ScoreInput
from cbce.scoring.encoder import (
    BASE_MODEL,
    BASE_REVISION,
    CLASSES,
    EncoderScorer,
    TrainConfig,
    fit,
)
from cbce.store.schema import Stance, Topic

H, N, D = Stance.HAWKISH, Stance.NEUTRAL, Stance.DOVISH


@pytest.mark.parametrize(
    ("probabilities", "stance", "confidence"),
    [
        pytest.param([0.7, 0.2, 0.1], H, 0.7, id="hawkish"),
        pytest.param([0.1, 0.8, 0.1], N, 0.8, id="neutral"),
        pytest.param([0.1, 0.2, 0.7], D, 0.7, id="dovish"),
        pytest.param([0.4, 0.4, 0.2], H, 0.4, id="tie_goes_to_the_lower_class_index"),
    ],
)
def test_scorer_reads_the_class_order_hawkish_neutral_dovish(
    probabilities: list[float], stance: Stance, confidence: float
) -> None:
    # Arrange
    scorer = EncoderScorer(lambda texts: [probabilities for _ in texts])

    # Act
    (score,) = scorer.score([ScoreInput("s", "text")], "run-1")

    # Assert
    assert score.stance is stance
    assert score.confidence == pytest.approx(confidence)
    assert (score.scorer, score.topic, score.run_id) == ("encoder", Topic.OTHER, "run-1")


def test_class_order_is_the_documented_one() -> None:
    # Arrange / Act / Assert
    assert CLASSES == (H, N, D)


def test_empty_input_never_calls_the_model() -> None:
    # Arrange
    scorer = EncoderScorer(lambda texts: pytest.fail("model should not be called"))

    # Act / Assert
    assert scorer.score([], "r") == []


@pytest.mark.parametrize(
    ("texts", "stances"),
    [
        pytest.param([], [], id="nothing_to_train_on"),
        pytest.param(["a"], [H, N], id="length_mismatch"),
    ],
)
def test_fit_rejects_bad_training_data(texts: list[str], stances: list[Stance]) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        fit(texts, stances)


tokenizer_cached = try_to_load_from_cache(BASE_MODEL, "tokenizer.json", revision=BASE_REVISION)
needs_tokenizer = pytest.mark.skipif(
    not isinstance(tokenizer_cached, str), reason="pinned DistilRoBERTa tokenizer not cached"
)


@pytest.fixture
def tiny_model(tmp_path: Path) -> str:
    """A seeded, randomly initialised 2-layer RoBERTa with the real tokenizer.

    Narrower networks sometimes stall on three classes for an unlucky seed; this size reached
    near-perfect fit on all five seeds tried.
    """
    import torch
    from transformers import RobertaConfig, RobertaForSequenceClassification

    torch.manual_seed(1234)  # without this each run starts from a different random network
    directory = tmp_path / "tiny"
    RobertaForSequenceClassification(
        RobertaConfig(
            vocab_size=50265,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=514,
            num_labels=3,
        )
    ).save_pretrained(directory)
    snapshot = Path(tokenizer_cached).parent
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy(snapshot / name, directory / name)
    return str(directory)


TEXTS = [
    "rates will rise sharply",
    "the committee will tighten policy",
    "inflation is rising fast",
    "policy remains on hold",
    "the meeting was held on tuesday",
    "no change in the stance",
    "growth is weak and rates will fall",
    "the committee will ease policy",
    "downside risks have increased",
]
STANCES = [H, H, H, N, N, N, D, D, D]
FAST = TrainConfig(epochs=2, batch_size=4, learning_rate=5e-3)


@needs_tokenizer
def test_training_returns_a_scorer_that_predicts_a_valid_distribution(tiny_model: str) -> None:
    # Arrange / Act
    scorer = fit(TEXTS, STANCES, FAST, base=tiny_model, revision=None)
    scores = scorer.score([ScoreInput(f"s{i}", t) for i, t in enumerate(TEXTS)], "r")

    # Assert
    assert [s.sentence_id for s in scores] == [f"s{i}" for i in range(len(TEXTS))]
    assert all(0.33 <= s.confidence <= 1.0 for s in scores)
    assert all(s.stance in CLASSES for s in scores)


@needs_tokenizer
def test_training_with_the_same_seed_is_reproducible(tiny_model: str) -> None:
    # Arrange
    items = [ScoreInput(f"s{i}", t) for i, t in enumerate(TEXTS)]

    # Act
    first = fit(TEXTS, STANCES, FAST, base=tiny_model, revision=None).score(items, "r")
    second = fit(TEXTS, STANCES, FAST, base=tiny_model, revision=None).score(items, "r")

    # Assert
    assert [s.stance for s in first] == [s.stance for s in second]
    assert [s.confidence for s in first] == pytest.approx([s.confidence for s in second], abs=1e-4)


@needs_tokenizer
def test_training_raises_the_probability_of_the_true_class(tiny_model: str) -> None:
    # Arrange: an untrained 3-class model gives each class about 1/3, a loss of ln 3
    config = TrainConfig(epochs=100, batch_size=9, learning_rate=1e-2, warmup_fraction=0.0)

    # Act
    scorer = fit(TEXTS, STANCES, config, base=tiny_model, revision=None)
    probabilities = scorer._predict(TEXTS)

    # Assert: it memorises nine sentences, so the true class gets nearly all the probability
    p_true = [probabilities[i][CLASSES.index(STANCES[i])] for i in range(len(TEXTS))]
    assert sum(p_true) / len(p_true) > 0.8


@needs_tokenizer
def test_a_different_seed_changes_the_trained_model(tiny_model: str) -> None:
    # Arrange
    items = [ScoreInput("s", "the committee will tighten policy")]

    # Act
    a = fit(
        TEXTS, STANCES, TrainConfig(epochs=2, batch_size=4, seed=0), base=tiny_model, revision=None
    )
    b = fit(
        TEXTS, STANCES, TrainConfig(epochs=2, batch_size=4, seed=1), base=tiny_model, revision=None
    )

    # Assert
    assert a.score(items, "r")[0].confidence != pytest.approx(b.score(items, "r")[0].confidence)
