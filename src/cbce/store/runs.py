"""Run records: tie every reported number to a git commit and a config hash (hard rule 8).

A commit hash only identifies code that is committed, so a run is refused unless the working
tree is clean. Untracked files that are not gitignored count as changes too, because a new
source file can change results just as an edited one can.
"""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from cbce.store.schema import Run


class DirtyTreeError(RuntimeError):
    """The code that would produce a run is not the code at the recorded commit."""


@dataclass(frozen=True)
class GitState:
    sha: str
    dirty: bool


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def git_state(repo: Path) -> GitState:
    return GitState(
        sha=_git(repo, "rev-parse", "HEAD"),
        dirty=bool(_git(repo, "status", "--porcelain")),
    )


def canonical_json(value: Any) -> str:
    """Stable text for hashing: sorted keys, no whitespace, and no silent type coercion."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def new_run_id(sha: str, started_at: datetime) -> str:
    return f"run-{started_at:%Y%m%dT%H%M%SZ}-{sha[:7]}"


def record_run(
    conn: duckdb.DuckDBPyConnection,
    repo: Path,
    run_id: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    started_at: datetime,
    expected_sha: str,
) -> Run:
    """Write one row to `runs`. Refuses if the tree is dirty or HEAD moved since the run began."""
    state = git_state(repo)
    if state.dirty:
        raise DirtyTreeError("uncommitted changes: commit them, then rerun to record this run")
    if state.sha != expected_sha:
        raise DirtyTreeError(
            f"HEAD moved from {expected_sha[:7]} to {state.sha[:7]} during the run"
        )
    run = Run(
        run_id=run_id,
        git_sha=state.sha,
        config_hash=config_hash(config),
        started_at=started_at,
        metrics_json=canonical_json({"config": config, "metrics": metrics}),
    )
    conn.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
        [run.run_id, run.git_sha, run.config_hash, run.started_at, run.metrics_json],
    )
    return run
