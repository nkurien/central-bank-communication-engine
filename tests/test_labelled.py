import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cbce.eval.labelled import LABELLED_FILES, load_labelled
from cbce.store import raw
from cbce.store.schema import Stance

# year 1999 and 2000 are dev years; 1998 and 2002 are test years
ROWS = {
    "train": [(0, "a", 1999, 1), (1, "b", 1998, 0), (2, "c", 2000, 2)],
    "test": [(0, "d", 2002, 1), (1, "e", 1999, 0)],
}


def build_store(root: Path, rows: dict[str, list[tuple]] = ROWS) -> dict[str, str]:
    files = {}
    for source, source_rows in rows.items():
        index, sentence, year, label = zip(*source_rows, strict=True)
        table = pa.table({"index": index, "sentence": sentence, "year": year, "label": label})
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        files[source] = raw.put(root, sink.getvalue().to_pybytes())[0]
    return files


@pytest.fixture
def files(tmp_path: Path) -> dict[str, str]:
    return build_store(tmp_path)


def test_dev_split_contains_only_dev_years_from_both_source_files(
    tmp_path: Path, files: dict[str, str]
) -> None:
    # Arrange: the authors' train and test files are pooled and re-split by year

    # Act
    dev = load_labelled(tmp_path, "dev", files=files)

    # Assert
    assert [s.sentence_id for s in dev] == ["tdw-test-1", "tdw-train-0", "tdw-train-2"]
    assert {s.text for s in dev} == {"a", "c", "e"}
    assert {s.year for s in dev} == {1999, 2000}


def test_test_split_is_refused_without_a_frozen_run(tmp_path: Path, files: dict[str, str]) -> None:
    # Arrange: no run id, or an empty one

    # Act / Assert
    for run_id in (None, ""):
        with pytest.raises(PermissionError):
            load_labelled(tmp_path, "test", frozen_run_id=run_id, files=files)


def test_test_split_loads_for_a_named_frozen_run(tmp_path: Path, files: dict[str, str]) -> None:
    # Arrange / Act
    test = load_labelled(tmp_path, "test", frozen_run_id="run-frozen-1", files=files)

    # Assert
    assert [s.sentence_id for s in test] == ["tdw-test-0", "tdw-train-1"]
    assert {s.text for s in test} == {"b", "d"}
    assert {s.year for s in test} == {1998, 2002}


def test_dev_and_test_never_share_a_sentence_or_a_year(
    tmp_path: Path, files: dict[str, str]
) -> None:
    # Arrange / Act
    dev = load_labelled(tmp_path, "dev", files=files)
    test = load_labelled(tmp_path, "test", frozen_run_id="r", files=files)

    # Assert
    assert not {s.sentence_id for s in dev} & {s.sentence_id for s in test}
    assert not {s.year for s in dev} & {s.year for s in test}
    assert len(dev) + len(test) == 5


@pytest.mark.parametrize(
    ("label", "expected"),
    [(0, Stance.DOVISH), (1, Stance.HAWKISH), (2, Stance.NEUTRAL)],
)
def test_label_codes_map_to_stances(tmp_path: Path, label: int, expected: Stance) -> None:
    # Arrange
    files = build_store(tmp_path, {"train": [(0, "x", 1999, label)]})

    # Act
    (sentence,) = load_labelled(tmp_path, "dev", files=files)

    # Assert
    assert sentence.stance is expected


def test_unknown_label_raises(tmp_path: Path) -> None:
    # Arrange
    files = build_store(tmp_path, {"train": [(0, "x", 1999, 7)]})

    # Act / Assert
    with pytest.raises(ValueError, match="unknown label"):
        load_labelled(tmp_path, "dev", files=files)


def test_sentence_ids_are_unique_across_files_with_overlapping_indexes(
    tmp_path: Path, files: dict[str, str]
) -> None:
    # Arrange: both files have an index 0 and an index 1

    # Act
    dev = load_labelled(tmp_path, "dev", files=files)

    # Assert
    ids = [s.sentence_id for s in dev]
    assert len(set(ids)) == len(ids)
    assert "tdw-train-0" in ids


def test_a_tampered_file_is_rejected(tmp_path: Path, files: dict[str, str]) -> None:
    # Arrange
    path = raw.raw_path(tmp_path, files["train"])
    path.chmod(0o644)
    path.write_bytes(b"not the original")

    # Act / Assert
    with pytest.raises(RuntimeError):
        load_labelled(tmp_path, "dev", files=files)


@pytest.mark.parametrize("source", sorted(LABELLED_FILES))
def test_pinned_digests_are_well_formed_sha256(source: str) -> None:
    # Arrange: digests were copied by hand from a shasum run, so check their shape

    # Act
    digest = LABELLED_FILES[source]

    # Assert
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


def test_loading_does_not_leave_the_interpreter_unable_to_exit(
    tmp_path: Path, files: dict[str, str]
) -> None:
    # Arrange: pq.read_table on an in-memory buffer once hung the process at exit, which
    # a test running inside pytest cannot see, so load in a fresh interpreter
    script = (
        "from pathlib import Path\n"
        "from cbce.eval.labelled import load_labelled\n"
        f"files = {files!r}\n"
        f"print(len(load_labelled(Path({str(tmp_path)!r}), 'dev', files=files)))\n"
    )

    # Act: raises TimeoutExpired if the interpreter never exits
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=True
    )

    # Assert
    assert result.stdout.strip() == "3"


def test_repeated_index_values_do_not_make_sentence_ids_collide(tmp_path: Path) -> None:
    # Arrange: the real files repeat `index` within one file
    rows = {"train": [(5, "a", 1999, 1), (5, "b", 2000, 0), (5, "c", 1999, 2)], "test": []}
    store = build_store(tmp_path, {"train": rows["train"]})

    # Act
    dev = load_labelled(tmp_path, "dev", files=store)

    # Assert
    assert [s.sentence_id for s in dev] == ["tdw-train-0", "tdw-train-1", "tdw-train-2"]
    assert [s.text for s in dev] == ["a", "b", "c"]
