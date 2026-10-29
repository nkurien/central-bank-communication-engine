from pathlib import Path

import pytest

from cbce.parse.segment import SPLITTER_VERSION, segment, sentence_id, split_sentences
from cbce.sources.fed.statement import parse_statement

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("paragraph", "expected"),
    [
        pytest.param("One. Two. Three.", ["One.", "Two.", "Three."], id="plain"),
        pytest.param("No terminal punctuation", ["No terminal punctuation"], id="no_punctuation"),
        pytest.param("", [], id="empty"),
        pytest.param("   \n  ", [], id="whitespace_only"),
        pytest.param(
            "Rates rose.   The   Committee\nagreed.",
            ["Rates rose.", "The Committee agreed."],
            id="whitespace_normalised",
        ),
        pytest.param(
            "The target is 5-1/4 to 5-1/2 percent. Inflation is 2.5 percent.",
            ["The target is 5-1/4 to 5-1/2 percent.", "Inflation is 2.5 percent."],
            id="fractions_and_decimals",
        ),
        pytest.param(
            "The U.S. economy grew. Mr. Powell spoke.",
            ["The U.S. economy grew.", "Mr. Powell spoke."],
            id="dotted_acronym_and_title",
        ),
        pytest.param(
            "Voted: Ben S. Bernanke, Chairman. Others agreed.",
            ["Voted: Ben S. Bernanke, Chairman.", "Others agreed."],
            id="middle_initial",
        ),
        pytest.param(
            "It met on Jan. 31. Nothing changed.",
            ["It met on Jan. 31.", "Nothing changed."],
            id="month_abbreviation_before_digit",
        ),
        pytest.param(
            "It met again in Mar. Nothing changed.",
            ["It met again in Mar. Nothing changed."],
            id="known_limit_abbreviation_before_capital_does_not_split",
        ),
        pytest.param(
            'Officials said "rates stay." Markets moved.',
            ['Officials said "rates stay."', "Markets moved."],
            id="closing_quote",
        ),
        pytest.param(
            "Is it done? Yes! It is.", ["Is it done?", "Yes!", "It is."], id="question_and_exclaim"
        ),
        pytest.param(
            "Inflation eased; however, it stays elevated.",
            ["Inflation eased; however, it stays elevated."],
            id="semicolon_is_not_a_boundary",
        ),
        pytest.param(
            "The U.S. Treasury sold bonds.",
            ["The U.S. Treasury sold bonds."],
            id="acronym_followed_by_capital",
        ),
    ],
)
def test_split_sentences(paragraph: str, expected: list[str]) -> None:
    # Arrange: a paragraph as the parser would return it

    # Act
    result = split_sentences(paragraph)

    # Assert
    assert result == expected


def test_segment_real_statement_gives_expected_sentence_counts_per_paragraph() -> None:
    # Arrange: paragraphs from the saved 31 January 2024 statement
    html = (FIXTURES / "fomc_statement_2024-01-31.html").read_text(encoding="utf-8")
    paragraphs = parse_statement(html).paragraphs

    # Act
    sentences = segment("a" * 64, paragraphs)

    # Assert
    per_paragraph = [sum(s.section == f"p{i}" for s in sentences) for i in range(len(paragraphs))]
    assert per_paragraph == [3, 3, 5, 3]
    assert [s.position for s in sentences] == list(range(14))
    assert sentences[0].text.startswith("Recent indicators suggest")
    assert all(s.text.endswith(".") for s in sentences)


@pytest.mark.parametrize("doc_id", ["a" * 64, "b" * 64])
def test_segment_ids_are_unique_stable_and_versioned(doc_id: str) -> None:
    # Arrange
    paragraphs = ["One. Two.", "Three."]

    # Act
    first = segment(doc_id, paragraphs)
    second = segment(doc_id, paragraphs)

    # Assert
    ids = [s.sentence_id for s in first]
    assert ids == [s.sentence_id for s in second]
    assert len(set(ids)) == 3
    assert all(SPLITTER_VERSION in i and i.startswith(doc_id[:12]) for i in ids)


@pytest.mark.parametrize(("doc_a", "doc_b"), [("a" * 64, "b" * 64)], ids=["different_documents"])
def test_same_position_in_different_documents_gets_different_ids(doc_a: str, doc_b: str) -> None:
    # Arrange / Act
    id_a, id_b = sentence_id(doc_a, 0), sentence_id(doc_b, 0)

    # Assert
    assert id_a != id_b


@pytest.mark.parametrize("paragraphs", [[], [""], ["", "  "]])
def test_segment_of_empty_input_is_empty(paragraphs: list[str]) -> None:
    # Arrange: documents with no text

    # Act
    result = segment("a" * 64, paragraphs)

    # Assert
    assert result == []
