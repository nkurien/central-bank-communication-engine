import json
import subprocess
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cbce.eval.dev_report import (
    build_metrics,
    cached_encoder_predictions,
    dev_config,
    eval_dev,
    format_table,
)
from cbce.eval.labelled import LabelledSentence
from cbce.scoring import encoder
from cbce.scoring.majority import MajorityScorer
from cbce.store import raw
from cbce.store.ddl import create_schema
from cbce.store.runs import DirtyTreeError, config_hash
from cbce.store.schema import Stance

H, N, D = Stance.HAWKISH, Stance.NEUTRAL, Stance.DOVISH
CYCLE = [H, N, D]


def sentences(n: int = 30) -> list[LabelledSentence]:
    # Every year holds every class, so a resample of years never loses a class.
    return [
        LabelledSentence(f"s{i}", f"text {i}", 1996 + i % 6, CYCLE[(i // 6) % 3]) for i in range(n)
    ]


def test_metrics_report_each_scorer_with_an_interval_and_per_class_detail() -> None:
    # Arrange: one perfect scorer, one always neutral
    data = sentences(36)
    truth = [s.stance for s in data]
    years = [s.year for s in data]
    predicted = {"perfect": list(truth), "neutral": [N] * len(truth)}

    # Act
    metrics = build_metrics(truth, years, predicted, n_resamples=100)

    # Assert
    assert metrics["n_sentences"] == 36 and metrics["n_year_clusters"] == 6
    perfect, neutral = metrics["scorers"]["perfect"], metrics["scorers"]["neutral"]
    assert perfect["macro_f1"] == 1.0 and perfect["ci_low"] == perfect["ci_high"] == 1.0
    assert neutral["macro_f1"] < 0.3
    assert set(perfect["per_class"]) == {"hawkish", "neutral", "dovish"}
    assert metrics["differences"]["neutral - perfect"]["estimate"] < 0


def test_metrics_are_json_serialisable() -> None:
    # Arrange: they are stored as JSON in the runs table
    data = sentences()
    truth = [s.stance for s in data]

    # Act
    metrics = build_metrics(
        truth, [s.year for s in data], {"a": truth, "b": [N] * len(truth)}, n_resamples=50
    )

    # Assert
    assert json.loads(json.dumps(metrics)) == metrics


def test_table_lists_scorers_best_first_and_marks_which_differences_exclude_zero() -> None:
    # Arrange
    data = sentences(36)
    truth = [s.stance for s in data]
    metrics = build_metrics(
        truth,
        [s.year for s in data],
        {"perfect": list(truth), "neutral": [N] * len(truth)},
        n_resamples=100,
    )

    # Act
    table = format_table(metrics)

    # Assert
    assert table.index("perfect") < table.index("neutral")
    assert "excludes 0" in table
    assert "n=36" in table


def fake_fit(calls: list[int]):
    def fit(train):
        calls.append(len(train))
        return MajorityScorer.fit(s.stance for s in train)

    return fit


def test_encoder_predictions_are_trained_once_then_read_from_the_cache(tmp_path: Path) -> None:
    # Arrange
    data, calls = sentences(), []

    # Act
    first, first_cached = cached_encoder_predictions(data, tmp_path, "r", fit=fake_fit(calls))
    second, second_cached = cached_encoder_predictions(data, tmp_path, "r", fit=fake_fit(calls))

    # Assert
    assert (first_cached, second_cached) == (False, True)
    assert first == second and len(first) == len(data)
    assert len(calls) == 5  # five folds, trained only on the first call


@pytest.mark.parametrize(
    ("change", "label"),
    [
        pytest.param(lambda d: d[:-1], "fewer_sentences", id="fewer_sentences"),
        pytest.param(
            lambda d: [LabelledSentence(s.sentence_id, s.text + "!", s.year, s.stance) for s in d],
            "edited_text",
            id="edited_text",
        ),
        pytest.param(
            lambda d: [LabelledSentence(s.sentence_id, s.text, s.year, N) for s in d],
            "relabelled",
            id="relabelled",
        ),
    ],
)
def test_changed_data_forces_retraining(tmp_path: Path, change, label: str) -> None:
    # Arrange: a populated cache
    data, calls = sentences(), []
    cached_encoder_predictions(data, tmp_path, "r", fit=fake_fit(calls))
    calls.clear()

    # Act
    _, from_cache = cached_encoder_predictions(change(data), tmp_path, "r", fit=fake_fit(calls))

    # Assert
    assert from_cache is False and len(calls) == 5


def test_changed_training_settings_force_retraining(tmp_path: Path) -> None:
    # Arrange
    data, calls = sentences(), []
    cached_encoder_predictions(data, tmp_path, "r", fit=fake_fit(calls))
    calls.clear()

    # Act
    _, from_cache = cached_encoder_predictions(
        data, tmp_path, "r", config=encoder.TrainConfig(learning_rate=1e-3), fit=fake_fit(calls)
    )

    # Assert
    assert from_cache is False and len(calls) == 5


def test_run_config_hash_changes_with_the_scorer_set() -> None:
    # Arrange / Act
    with_encoder, without = dev_config(), dev_config(include_encoder=False)

    # Assert
    assert "encoder_cv" in with_encoder["scorers"] and "encoder_cv" not in without["scorers"]
    assert config_hash(with_encoder) != config_hash(without)
    assert config_hash(dev_config()) == config_hash(with_encoder)


# --- the orchestrator, against a throwaway git repo and a synthetic labelled store ---------


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "t@example.test")
    git(path, "config", "user.name", "T")
    (path / "code.py").write_text("x = 1\n")
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "initial")
    return path


@pytest.fixture
def labelled(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "raw"  # outside the repo, so it never makes the tree dirty
    rows = [(i, f"sentence {i}", 1996 + i % 6, i % 3) for i in range(30)]  # years 1998 are test
    index, text, year, label = zip(*rows, strict=True)
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"index": index, "sentence": text, "year": year, "label": label}), sink)
    digest = raw.put(root, sink.getvalue().to_pybytes())[0]
    return root, {"train": digest}


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    create_schema(connection)
    return connection


def make_predict(seen: list[tuple[int, str]]):
    def predict(dev, run_id):
        seen.append((len(dev), run_id))
        truth = [s.stance for s in dev]
        return {"perfect": list(truth), "neutral": [N] * len(truth)}

    return predict


def run_rows(conn: duckdb.DuckDBPyConnection) -> list[tuple]:
    return conn.execute("SELECT run_id, git_sha, config_hash FROM runs").fetchall()


def test_clean_tree_records_a_run_whose_id_is_passed_to_the_scorers(
    repo: Path, labelled, conn: duckdb.DuckDBPyConnection
) -> None:
    # Arrange
    root, files = labelled
    seen: list[tuple[int, str]] = []
    config = {"split": "test"}

    # Act
    metrics, run = eval_dev(conn, repo, root, make_predict(seen), config, labelled_files=files)

    # Assert
    assert run is not None and run_rows(conn) == [(run.run_id, run.git_sha, config_hash(config))]
    assert seen == [(25, run.run_id)]  # 5 of 30 sentences fall in test year 1998
    assert metrics["scorers"]["perfect"]["macro_f1"] == 1.0
    stored = conn.execute("SELECT metrics_json FROM runs").fetchone()[0]
    assert json.loads(stored)["metrics"] == metrics


def test_dirty_tree_is_refused_before_any_scoring(
    repo: Path, labelled, conn: duckdb.DuckDBPyConnection
) -> None:
    # Arrange
    root, files = labelled
    (repo / "code.py").write_text("x = 2\n")
    seen: list[tuple[int, str]] = []

    # Act / Assert
    with pytest.raises(DirtyTreeError, match="--no-record"):
        eval_dev(conn, repo, root, make_predict(seen), {}, labelled_files=files)
    assert seen == [] and run_rows(conn) == []


def test_no_record_scores_a_dirty_tree_and_writes_nothing(
    repo: Path, labelled, conn: duckdb.DuckDBPyConnection
) -> None:
    # Arrange
    root, files = labelled
    (repo / "code.py").write_text("x = 2\n")
    seen: list[tuple[int, str]] = []

    # Act
    metrics, run = eval_dev(
        conn, repo, root, make_predict(seen), {}, record=False, labelled_files=files
    )

    # Assert
    assert run is None and run_rows(conn) == []
    assert len(seen) == 1 and "perfect" in metrics["scorers"]


def test_tree_dirtied_during_the_run_is_caught_at_recording(
    repo: Path, labelled, conn: duckdb.DuckDBPyConnection
) -> None:
    # Arrange: the scorers "run", and someone edits code while they do
    root, files = labelled

    def predict(dev, run_id):
        (repo / "code.py").write_text("x = 3\n")
        truth = [s.stance for s in dev]
        return {"perfect": list(truth)}

    # Act / Assert
    with pytest.raises(DirtyTreeError, match="uncommitted"):
        eval_dev(conn, repo, root, predict, {}, labelled_files=files)
    assert run_rows(conn) == []


def test_the_test_split_is_never_loaded(
    repo: Path, labelled, conn: duckdb.DuckDBPyConnection
) -> None:
    # Arrange: record every year the scorers are shown
    root, files = labelled
    years: set[int] = set()

    def predict(dev, run_id):
        years.update(s.year for s in dev)
        return {"perfect": [s.stance for s in dev]}

    # Act
    eval_dev(conn, repo, root, predict, {}, labelled_files=files)

    # Assert: 1998 and 2002 are test years under the frozen rule
    assert years == {1996, 1997, 1999, 2000, 2001}


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"model_version": "digest-2"}, id="weights"),
        pytest.param({"prompt_sha256": "def"}, id="prompt_text"),
        pytest.param({"seed": 1}, id="seed"),
    ],
)
def test_llm_settings_are_part_of_the_run_config_hash(changed: dict[str, object]) -> None:
    # Arrange
    base = {"model": "m", "model_version": "digest-1", "prompt_sha256": "abc", "seed": 0}

    # Act
    original = config_hash(dev_config(include_encoder=False, llm=base))
    altered = config_hash(dev_config(include_encoder=False, llm={**base, **changed}))

    # Assert
    assert original != altered
    assert original != config_hash(dev_config(include_encoder=False))
