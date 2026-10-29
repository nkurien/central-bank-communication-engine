"""Loader for the labelled FOMC sentences ("Trillion Dollar Words", CC BY-NC 4.0).

Hard rule 3: prompts are tuned on dev only. The test split cannot be loaded without naming
the frozen run that is allowed to read it, and dev loading never builds a test-year row.
"""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cbce.eval.splits import Split, split_for_year
from cbce.store import raw
from cbce.store.schema import Stance

# The authors' two files, as stored by content hash in data/raw. We pool them and re-split
# by year, because their own split is random per sentence and leaks across a meeting.
LABELLED_FILES = {
    "train": "51f20a6c5c61662b80b5e6ff675a0aa32eba9d7ca4080e4b06745d86e13c0c1e",
    "test": "3c9dbd7a3ab477c44e1be47281a9f9113c23fd39430c6d978175da1e46f4fcf4",
}

# From the dataset card: 0 Dovish, 1 Hawkish, 2 Neutral.
_LABEL_TO_STANCE = {0: Stance.DOVISH, 1: Stance.HAWKISH, 2: Stance.NEUTRAL}


@dataclass(frozen=True)
class LabelledSentence:
    sentence_id: str
    text: str
    year: int
    stance: Stance


def load_labelled(
    raw_root: Path,
    split: Split,
    *,
    frozen_run_id: str | None = None,
    files: dict[str, str] = LABELLED_FILES,
) -> list[LabelledSentence]:
    """Labelled sentences for one split, ordered by sentence_id."""
    if split == "test" and not frozen_run_id:
        raise PermissionError("the test split needs the run_id of a frozen prompt version")

    sentences: list[LabelledSentence] = []
    for source, digest in files.items():
        # ParquetFile, not pq.read_table: read_table on an in-memory buffer leaves a thread
        # pool behind that stops the interpreter from exiting.
        table = pq.ParquetFile(pa.BufferReader(raw.get(raw_root, digest))).read()
        rows = zip(
            table.column("index").to_pylist(),
            table.column("sentence").to_pylist(),
            table.column("year").to_pylist(),
            table.column("label").to_pylist(),
            strict=True,
        )
        for index, text, year, label in rows:
            if split_for_year(year) != split:
                continue
            if label not in _LABEL_TO_STANCE:
                raise ValueError(f"unknown label {label!r} at {source}[{index}]")
            sentences.append(
                LabelledSentence(f"tdw-{source}-{index}", text, year, _LABEL_TO_STANCE[label])
            )
    return sorted(sentences, key=lambda s: s.sentence_id)
