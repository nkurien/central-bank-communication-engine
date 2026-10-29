import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from cbce.store.ddl import create_schema
from cbce.store.runs import (
    DirtyTreeError,
    canonical_json,
    config_hash,
    git_state,
    new_run_id,
    record_run,
)

STARTED = datetime(2026, 10, 29, 22, 15, 3, tzinfo=UTC)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.test")
    git(path, "config", "user.name", "Test")
    (path / ".gitignore").write_text("ignored.txt\n")
    (path / "code.py").write_text("x = 1\n")
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "initial")
    return path


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    create_schema(connection)
    return connection


def run_count(conn: duckdb.DuckDBPyConnection) -> int:
    return conn.execute("SELECT count(*) FROM runs").fetchone()[0]


def test_a_fresh_commit_is_clean_and_reports_the_full_sha(repo: Path) -> None:
    # Arrange: a repo with one commit and nothing else

    # Act
    state = git_state(repo)

    # Assert
    assert state.dirty is False
    assert state.sha == git(repo, "rev-parse", "HEAD")
    assert len(state.sha) == 40


@pytest.mark.parametrize(
    ("change", "expected_dirty"),
    [
        pytest.param(
            lambda r: (r / "code.py").write_text("x = 2\n"), True, id="edited_tracked_file"
        ),
        pytest.param(lambda r: (r / "new.py").write_text("y = 1\n"), True, id="untracked_file"),
        pytest.param(
            lambda r: (
                (r / "staged.py").write_text("z = 1\n"),
                git(r, "add", "staged.py"),
            ),
            True,
            id="staged_but_uncommitted",
        ),
        pytest.param(
            lambda r: (r / "ignored.txt").write_text("noise\n"), False, id="gitignored_file"
        ),
    ],
)
def test_dirty_detection(repo: Path, change, expected_dirty: bool) -> None:
    # Arrange
    change(repo)

    # Act
    state = git_state(repo)

    # Assert
    assert state.dirty is expected_dirty


def test_committing_the_change_makes_the_tree_clean_again(repo: Path) -> None:
    # Arrange
    (repo / "code.py").write_text("x = 2\n")
    assert git_state(repo).dirty

    # Act
    git(repo, "commit", "-q", "-am", "change")

    # Assert
    assert git_state(repo).dirty is False


def test_config_hash_ignores_key_order_but_not_values() -> None:
    # Arrange
    a = {"split": "v1", "bootstrap": {"seed": 1, "n": 2}}
    reordered = {"bootstrap": {"n": 2, "seed": 1}, "split": "v1"}
    changed = {"split": "v1", "bootstrap": {"seed": 2, "n": 2}}

    # Act / Assert
    assert config_hash(a) == config_hash(reordered)
    assert config_hash(a) != config_hash(changed)


def test_canonical_json_refuses_values_it_would_have_to_guess_at() -> None:
    # Arrange: a set has no canonical JSON form

    # Act / Assert
    with pytest.raises(TypeError):
        canonical_json({"scorers": {"a", "b"}})


def test_run_id_is_readable_and_names_the_commit() -> None:
    # Arrange / Act
    run_id = new_run_id("abcdef1234567890", STARTED)

    # Assert
    assert run_id == "run-20261029T221503Z-abcdef1"


def test_record_run_stores_a_traceable_row(repo: Path, conn: duckdb.DuckDBPyConnection) -> None:
    # Arrange
    sha = git_state(repo).sha
    config = {"split": "v1"}
    metrics = {"macro_f1": 0.5}

    # Act
    run = record_run(conn, repo, "run-1", config, metrics, STARTED, sha)

    # Assert
    row = conn.execute("SELECT run_id, git_sha, config_hash, metrics_json FROM runs").fetchone()
    assert row[:3] == ("run-1", sha, config_hash(config))
    assert json.loads(row[3]) == {"config": config, "metrics": metrics}
    assert run.git_sha == sha


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda r: (r / "code.py").write_text("x = 2\n"), id="edited_file"),
        pytest.param(lambda r: (r / "new.py").write_text("y = 1\n"), id="new_file"),
    ],
)
def test_record_run_refuses_a_dirty_tree_and_writes_nothing(
    repo: Path, conn: duckdb.DuckDBPyConnection, change
) -> None:
    # Arrange
    sha = git_state(repo).sha
    change(repo)

    # Act / Assert
    with pytest.raises(DirtyTreeError, match="uncommitted"):
        record_run(conn, repo, "run-1", {}, {}, STARTED, sha)
    assert run_count(conn) == 0


def test_record_run_refuses_when_head_moved_during_the_run(
    repo: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    # Arrange: the run began at one commit, then another was made before recording
    started_at_sha = git_state(repo).sha
    (repo / "code.py").write_text("x = 2\n")
    git(repo, "commit", "-q", "-am", "moved on")

    # Act / Assert
    with pytest.raises(DirtyTreeError, match="HEAD moved"):
        record_run(conn, repo, "run-1", {}, {}, STARTED, started_at_sha)
    assert run_count(conn) == 0


def test_the_same_run_id_cannot_be_recorded_twice(
    repo: Path, conn: duckdb.DuckDBPyConnection
) -> None:
    # Arrange
    sha = git_state(repo).sha
    record_run(conn, repo, "run-1", {}, {}, STARTED, sha)

    # Act / Assert
    with pytest.raises(duckdb.ConstraintException):
        record_run(conn, repo, "run-1", {}, {}, STARTED, sha)
