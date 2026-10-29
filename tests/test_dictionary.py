import hashlib
from pathlib import Path

import pytest

from cbce.scoring.base import ScoreInput
from cbce.scoring.dictionary import (
    LM_MASTER_DICTIONARY_SHA256,
    LMToneScorer,
    WordListScorer,
    compile_terms,
    decide,
)
from cbce.store.schema import Stance, Topic

H, N, D = Stance.HAWKISH, Stance.NEUTRAL, Stance.DOVISH
HAND_LIST = Path(__file__).parent.parent / "config" / "dictionaries" / "hand_list_v1.txt"
HAND_LIST_V1_SHA256 = "96ae6bdb34a48dc7ec12c8c0aec405ba9be0f53c8e475c00d269bde120aa1046"


@pytest.mark.parametrize(
    ("up", "down", "stance", "confidence"),
    [
        pytest.param(3, 1, H, 0.5, id="hawkish_wins"),
        pytest.param(1, 3, D, 0.5, id="dovish_wins"),
        pytest.param(2, 2, N, 0.0, id="tie_is_neutral"),
        pytest.param(0, 0, N, 0.0, id="no_hits_is_neutral"),
        pytest.param(4, 0, H, 1.0, id="unanimous_has_full_margin"),
    ],
)
def test_decide(up: int, down: int, stance: Stance, confidence: float) -> None:
    # Arrange: counts of up and down hits

    # Act
    result = decide(up, down)

    # Assert
    assert result == (stance, confidence)


@pytest.mark.parametrize(
    ("terms", "text", "expected_hits"),
    [
        pytest.param(
            ["tighten*"], "The Fed will tighten and tightening continues.", 2, id="wildcard"
        ),
        pytest.param(
            ["firming"], "Firms are hiring; firming is possible.", 1, id="whole_word_only"
        ),
        pytest.param(["cut"], "A cutoff and a cut and cuts.", 1, id="no_partial_word_match"),
        pytest.param(
            ["raise the target"], "To RAISE the   target range.", 1, id="phrase_any_case_and_space"
        ),
        pytest.param(
            ["rate*", "higher rate*"], "higher rates and rate", 2, id="longer_phrase_wins"
        ),
        pytest.param(["strong*"], "", 0, id="empty_text"),
    ],
)
def test_compile_terms_matching(terms: list[str], text: str, expected_hits: int) -> None:
    # Arrange
    pattern = compile_terms(terms)

    # Act
    hits = pattern.findall(text)

    # Assert
    assert len(hits) == expected_hits


def test_compile_terms_rejects_an_empty_list() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        compile_terms([])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Policy should tighten and inflation pressures rise.", H, id="hawkish_hits"),
        pytest.param("Growth is weak and policy remains accommodative.", D, id="dovish_hits"),
        pytest.param("The Committee met on Tuesday.", N, id="no_hits"),
        pytest.param("Strong growth but weak hiring.", N, id="balanced_hits"),
    ],
)
def test_word_list_scorer_decisions(text: str, expected: Stance) -> None:
    # Arrange
    scorer = WordListScorer(
        "hand_list_test",
        hawkish=["tighten*", "inflation pressure*", "strong*"],
        dovish=["weak*", "accommodative"],
    )

    # Act
    (score,) = scorer.score([ScoreInput("s1", text)], "run-1")

    # Assert
    assert score.stance is expected
    assert (score.scorer, score.topic, score.run_id) == ("hand_list_test", Topic.OTHER, "run-1")


@pytest.mark.parametrize("count", [0, 1, 4])
def test_scorers_return_one_record_per_input_in_order(count: int) -> None:
    # Arrange
    items = [ScoreInput(f"s{i}", "weak and slow") for i in range(count)]
    scorer = WordListScorer("w", hawkish=["strong"], dovish=["weak"])

    # Act
    scores = scorer.score(items, "r")

    # Assert
    assert [s.sentence_id for s in scores] == [item.sentence_id for item in items]


LM_CSV = (
    "Word,Seq_num,Negative,Positive,Uncertainty\n"
    "GAIN,1,0,2009,0\n"
    "LOSS,2,2009,0,0\n"
    "REMOVEDNEG,3,-2020,0,0\n"
    "REMOVEDPOS,4,0,-2020,0\n"
    "MAYBE,5,0,0,2009\n"
    "NEUTRALWORD,6,0,0,0\n"
)


def test_lm_csv_uses_only_words_currently_flagged() -> None:
    # Arrange: negative years mean the word was removed from the category

    # Act
    scorer = LMToneScorer.from_csv(LM_CSV.encode())

    # Assert
    assert scorer.positive == {"gain"}
    assert scorer.negative == {"loss"}


def test_lm_csv_handles_a_byte_order_mark() -> None:
    # Arrange
    content = b"\xef\xbb\xbf" + LM_CSV.encode()

    # Act
    scorer = LMToneScorer.from_csv(content)

    # Assert
    assert scorer.positive == {"gain"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Gain after gain.", H, id="positive_words_read_as_hawkish"),
        pytest.param("A loss, then a LOSS.", D, id="negative_words_read_as_dovish"),
        pytest.param("Gain and loss.", N, id="tie"),
        pytest.param("Nothing here.", N, id="no_hits"),
        pytest.param("Gains and losses.", N, id="inflected_forms_are_not_stemmed"),
    ],
)
def test_lm_tone_scorer(text: str, expected: Stance) -> None:
    # Arrange
    scorer = LMToneScorer.from_csv(LM_CSV.encode())

    # Act
    (score,) = scorer.score([ScoreInput("s", text)], "r")

    # Assert
    assert score.stance is expected
    assert score.scorer == "lm_tone"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        pytest.param("neutral\tfoo", "expected", id="unknown_class"),
        pytest.param("hawkish", "expected", id="missing_term"),
        pytest.param("hawkish\t  ", "expected", id="blank_term"),
    ],
)
def test_word_list_file_rejects_malformed_lines(tmp_path: Path, line: str, message: str) -> None:
    # Arrange
    path = tmp_path / "list.txt"
    path.write_text(f"# comment\n\nhawkish\ttighten*\n{line}\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match=message):
        WordListScorer.from_file(path, "x")


def test_hand_list_v1_is_frozen() -> None:
    # Arrange: a changed list is a new version file, never an edit in place

    # Act
    digest = hashlib.sha256(HAND_LIST.read_bytes()).hexdigest()

    # Assert
    assert digest == HAND_LIST_V1_SHA256


def test_hand_list_v1_loads_and_separates_the_two_directions() -> None:
    # Arrange
    scorer = WordListScorer.from_file(HAND_LIST, "hand_list_v1")
    items = [
        ScoreInput("a", "The Committee decided to tighten policy."),
        ScoreInput("b", "Activity was weak and policy stays accommodative."),
        ScoreInput("c", "The meeting was held on Tuesday."),
    ]

    # Act
    stances = [s.stance for s in scorer.score(items, "r")]

    # Assert
    assert stances == [H, D, N]


def test_pinned_lm_digest_is_a_well_formed_sha256() -> None:
    # Arrange: copied by hand from a hashing run, so check its shape

    # Act
    digest = LM_MASTER_DICTIONARY_SHA256

    # Assert
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
