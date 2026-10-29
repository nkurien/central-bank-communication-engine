"""Split document paragraphs into sentences with stable IDs.

The splitter is rule-based and deterministic. Any change to its behaviour must bump
SPLITTER_VERSION, which is part of every sentence ID, so scores from different splitter
versions can never be confused.
"""

import re
from collections.abc import Sequence

from cbce.store.schema import Sentence

SPLITTER_VERSION = "v1"

# Words that end in a period without ending the sentence.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "jr", "sr", "st", "no", "nos", "vs", "etc", "inc", "corp", "co",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    }
)  # fmt: skip

# A sentence end: terminal punctuation, optional closing quote or bracket, then whitespace.
_BOUNDARY = re.compile(r"""([.!?]+["')\]]*)\s+""")
# What may start the next sentence.
_SENTENCE_START = re.compile(r"""["'(\[]?[A-Z0-9]""")


def _ends_abbreviation(before: str) -> bool:
    """True if the text before a period ends in an abbreviation, initial or dotted acronym."""
    token = before.split()[-1] if before.split() else ""
    bare = token.lstrip("\"'([")
    if "." in bare:  # U.S., p.m., a.m.
        return True
    if len(bare) == 1 and bare.isalpha() and bare.isupper():  # middle initial: "Ben S. Bernanke"
        return True
    return bare.lower() in _ABBREVIATIONS


def split_sentences(paragraph: str) -> list[str]:
    """Split one paragraph into sentences."""
    text = " ".join(paragraph.split())
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        following = text[end:]
        if not following or not _SENTENCE_START.match(following):
            continue
        if match.group(1).startswith(".") and _ends_abbreviation(text[start : match.start()]):
            continue
        sentences.append(text[start : match.end(1)].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def sentence_id(doc_id: str, position: int) -> str:
    return f"{doc_id[:12]}-{SPLITTER_VERSION}-{position:04d}"


def segment(doc_id: str, paragraphs: Sequence[str]) -> list[Sentence]:
    """Sentences for a document. `position` runs across the whole document; `section`
    records the zero-based paragraph the sentence came from, as 'p0', 'p1', ...
    """
    sentences: list[Sentence] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        for text in split_sentences(paragraph):
            position = len(sentences)
            sentences.append(
                Sentence(
                    sentence_id=sentence_id(doc_id, position),
                    doc_id=doc_id,
                    position=position,
                    text=text,
                    section=f"p{paragraph_index}",
                )
            )
    return sentences
