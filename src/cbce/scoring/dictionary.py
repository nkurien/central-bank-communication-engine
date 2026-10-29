"""Dictionary baselines: Loughran-McDonald tone and a hand-written hawkish/dovish word list.

Both count words that point up (hawkish) and down (dovish) in a sentence and decide by the
sign of the difference: hawkish if up wins, dovish if down wins, neutral on a tie or no hits.
The rule has no tuned threshold. Confidence is the winning margin, |up - down| / (up + down),
and 0 for neutral; dictionaries have no calibrated confidence, so it is only a rough ordering.
"""

import csv
import io
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from cbce.scoring.base import ScoreInput
from cbce.store.schema import Stance, StanceScore, Topic

_WORD = re.compile(r"[a-z]+")

# Loughran-McDonald Master Dictionary 1993-2025 as stored by content hash in data/raw. Free for
# academic research; cite the authors and never commit the file.
LM_MASTER_DICTIONARY_SHA256 = "e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d"


def decide(up: int, down: int) -> tuple[Stance, float]:
    total = up + down
    if up > down:
        return Stance.HAWKISH, (up - down) / total
    if down > up:
        return Stance.DOVISH, (down - up) / total
    return Stance.NEUTRAL, 0.0


def _score(
    name: str, items: Sequence[ScoreInput], run_id: str, counts: list[tuple[int, int]]
) -> list[StanceScore]:
    scores = []
    for item, (up, down) in zip(items, counts, strict=True):
        stance, confidence = decide(up, down)
        scores.append(
            StanceScore(
                sentence_id=item.sentence_id,
                scorer=name,
                topic=Topic.OTHER,
                stance=stance,
                confidence=confidence,
                run_id=run_id,
            )
        )
    return scores


def compile_terms(terms: Iterable[str]) -> re.Pattern[str]:
    """One whole-word, case-insensitive pattern. 'x*' matches any word starting with x, and
    multi-word terms match as phrases. Longer terms are tried first so phrases win.
    """
    parts = []
    for term in sorted({t.lower() for t in terms}, key=len, reverse=True):
        words = [
            re.escape(word[:-1]) + r"\w*" if word.endswith("*") else re.escape(word)
            for word in term.split()
        ]
        parts.append(r"\s+".join(words))
    if not parts:
        raise ValueError("no terms")
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


class WordListScorer:
    """Hawkish minus dovish word hits from a hand-written list."""

    def __init__(self, name: str, hawkish: Iterable[str], dovish: Iterable[str]) -> None:
        self.name = name
        self._hawkish = compile_terms(hawkish)
        self._dovish = compile_terms(dovish)

    @classmethod
    def from_file(cls, path: Path, name: str) -> "WordListScorer":
        """Read '<class><TAB><term>' lines; blank lines and '#' comments are skipped."""
        terms: dict[str, list[str]] = {"hawkish": [], "dovish": []}
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip() or line.startswith("#"):
                continue
            label, _, term = line.partition("\t")
            if label not in terms or not term.strip():
                raise ValueError(f"{path}:{number}: expected '<hawkish|dovish><TAB><term>'")
            terms[label].append(term.strip())
        return cls(name, terms["hawkish"], terms["dovish"])

    def score(self, items: Sequence[ScoreInput], run_id: str) -> list[StanceScore]:
        counts = [
            (len(self._hawkish.findall(item.text)), len(self._dovish.findall(item.text)))
            for item in items
        ]
        return _score(self.name, items, run_id, counts)


class LMToneScorer:
    """Loughran-McDonald positive minus negative words, read as hawkish minus dovish.

    The mapping is a hypothesis, not a fact: a finance-sentiment list was not built for
    monetary policy, and good news for the economy is hawkish for rates, so positive words
    point to hawkish. Whether that is any use is what the baseline measures.
    """

    name = "lm_tone"

    def __init__(self, positive: frozenset[str], negative: frozenset[str]) -> None:
        self.positive = positive
        self.negative = negative

    @classmethod
    def from_csv(cls, content: bytes) -> "LMToneScorer":
        """Words flagged in the Positive or Negative column.

        Those cells hold the year a word was added, and a negative year (e.g. -2020) means it
        was removed from the category that year, so only values above zero count.
        """
        rows = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        positive, negative = set(), set()
        for row in rows:
            word = row["Word"].lower()
            if int(row["Positive"]) > 0:
                positive.add(word)
            if int(row["Negative"]) > 0:
                negative.add(word)
        return cls(frozenset(positive), frozenset(negative))

    @classmethod
    def from_raw(cls, raw_root: Path) -> "LMToneScorer":
        from cbce.store import raw

        return cls.from_csv(raw.get(raw_root, LM_MASTER_DICTIONARY_SHA256))

    def score(self, items: Sequence[ScoreInput], run_id: str) -> list[StanceScore]:
        counts = []
        for item in items:
            words = _WORD.findall(item.text.lower())
            counts.append(
                (sum(w in self.positive for w in words), sum(w in self.negative for w in words))
            )
        return _score(self.name, items, run_id, counts)
