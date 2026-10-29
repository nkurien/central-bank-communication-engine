"""The dev-split evaluation table: every baseline on the same sentences, with year-clustered
intervals, recorded as one run. Tuning happens on dev only; the test split is never loaded here.
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from cbce.eval.bootstrap import BOOTSTRAP_SEED, bootstrap_macro_f1
from cbce.eval.crossval import out_of_fold_predictions
from cbce.eval.labelled import LABELLED_FILES, LabelledSentence, load_labelled
from cbce.eval.metrics import evaluate
from cbce.eval.splits import SPLIT_NAME
from cbce.scoring import encoder
from cbce.scoring.base import ScoreInput
from cbce.scoring.dictionary import LM_MASTER_DICTIONARY_SHA256, LMToneScorer, WordListScorer
from cbce.scoring.finbert import FINBERT_MODEL, FINBERT_REVISION, FinBertScorer
from cbce.scoring.majority import MajorityScorer
from cbce.store.runs import DirtyTreeError, canonical_json, git_state, new_run_id, record_run
from cbce.store.schema import Run, Stance

HAND_LIST = Path("config/dictionaries/hand_list_v1.txt")
N_RESAMPLES = 2000
CV_FOLDS = 5

Predict = Callable[[Sequence[LabelledSentence], str], dict[str, list[Stance]]]


def dev_config(*, include_encoder: bool = True) -> dict[str, Any]:
    """Everything that determines the table, so its hash identifies the experiment."""
    config: dict[str, Any] = {
        "split": SPLIT_NAME,
        "labelled_files": LABELLED_FILES,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "n_resamples": N_RESAMPLES, "level": 0.95},
        "scorers": {
            "majority": "most common dev label",
            "lm_tone": {"dictionary_sha256": LM_MASTER_DICTIONARY_SHA256},
            "hand_list_v1": {"file_sha256": hashlib.sha256(HAND_LIST.read_bytes()).hexdigest()},
            "finbert": {"model": FINBERT_MODEL, "revision": FINBERT_REVISION},
        },
    }
    if include_encoder:
        config["scorers"]["encoder_cv"] = {
            "base": encoder.BASE_MODEL,
            "revision": encoder.BASE_REVISION,
            "train": asdict(encoder.TrainConfig()),
            "folds": CV_FOLDS,
        }
    return config


def build_metrics(
    truth: Sequence[Stance],
    years: Sequence[int],
    predicted: Mapping[str, Sequence[Stance]],
    *,
    n_resamples: int = N_RESAMPLES,
) -> dict[str, Any]:
    per_scorer, differences = bootstrap_macro_f1(truth, predicted, years, n_resamples=n_resamples)
    scorers = {}
    for name, interval in per_scorer.items():
        report = evaluate(truth, predicted[name])
        scorers[name] = {
            "macro_f1": interval.estimate,
            "ci_low": interval.low,
            "ci_high": interval.high,
            "per_class": {
                label.value: {"precision": m.precision, "recall": m.recall, "f1": m.f1}
                for label, m in report.per_class.items()
            },
        }
    return {
        "n_sentences": len(truth),
        "n_year_clusters": len(set(years)),
        "scorers": scorers,
        "differences": {
            f"{a} - {b}": {"estimate": iv.estimate, "ci_low": iv.low, "ci_high": iv.high}
            for (a, b), iv in differences.items()
        },
    }


def format_table(metrics: Mapping[str, Any]) -> str:
    lines = [
        f"dev n={metrics['n_sentences']}, {metrics['n_year_clusters']} year-clusters, "
        "95% percentile CI",
    ]
    ranked = sorted(metrics["scorers"].items(), key=lambda kv: -kv[1]["macro_f1"])
    for name, m in ranked:
        lines.append(
            f"  {name:13s} macro-F1 {m['macro_f1']:.3f}  [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"
        )
    lines.append("paired differences (first minus second):")
    for pair, d in sorted(metrics["differences"].items()):
        verdict = "excludes 0" if d["ci_low"] > 0 or d["ci_high"] < 0 else "includes 0"
        interval = f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
        lines.append(f"  {pair:28s} {d['estimate']:+.3f}  {interval}  {verdict}")
    return "\n".join(lines)


def cached_encoder_predictions(
    dev: Sequence[LabelledSentence],
    cache_dir: Path,
    run_id: str,
    *,
    config: encoder.TrainConfig | None = None,
    fit: Callable[[Sequence[LabelledSentence]], Any] | None = None,
) -> tuple[list[Stance], bool]:
    """Out-of-fold encoder predictions, trained once per distinct (data, settings, model).

    Returns (predictions in dev order, whether they came from the cache). The key covers the
    base model revision, every training setting, the fold count and the exact sentences and
    labels, so any change retrains instead of silently reusing stale predictions.
    """
    config = config or encoder.TrainConfig()
    data_fingerprint = hashlib.sha256(
        canonical_json([[s.sentence_id, s.text, s.stance.value] for s in dev]).encode()
    ).hexdigest()
    key = hashlib.sha256(
        canonical_json(
            {
                "base": encoder.BASE_MODEL,
                "revision": encoder.BASE_REVISION,
                "train": asdict(config),
                "folds": CV_FOLDS,
                "split": SPLIT_NAME,
                "data": data_fingerprint,
            }
        ).encode()
    ).hexdigest()[:16]
    path = cache_dir / f"encoder_oof_{key}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        return [Stance(cached[s.sentence_id]) for s in dev], True

    def default_fit(train: Sequence[LabelledSentence]) -> Any:
        return encoder.fit([s.text for s in train], [s.stance for s in train], config)

    scores = out_of_fold_predictions(dev, fit or default_fit, run_id, k=CV_FOLDS)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({s.sentence_id: s.stance.value for s in scores}))
    return [s.stance for s in scores], False


def real_predictions(raw_root: Path, cache_dir: Path, *, include_encoder: bool) -> Predict:
    """Predictions from every real scorer for the given dev sentences."""

    def predict(dev: Sequence[LabelledSentence], run_id: str) -> dict[str, list[Stance]]:
        items = [ScoreInput(s.sentence_id, s.text) for s in dev]
        scorers = [
            MajorityScorer.fit(s.stance for s in dev),
            LMToneScorer.from_raw(raw_root),
            WordListScorer.from_file(HAND_LIST, "hand_list_v1"),
            FinBertScorer.load(),
        ]
        predicted = {
            scorer.name: [p.stance for p in scorer.score(items, run_id)] for scorer in scorers
        }
        if include_encoder:
            predicted["encoder_cv"], _ = cached_encoder_predictions(dev, cache_dir, run_id)
        return predicted

    return predict


def eval_dev(
    conn: duckdb.DuckDBPyConnection,
    repo: Path,
    raw_root: Path,
    predict: Predict,
    config: Mapping[str, Any],
    *,
    record: bool = True,
    labelled_files: Mapping[str, str] = LABELLED_FILES,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[dict[str, Any], Run | None]:
    """Score every baseline on dev and, unless told not to, record the run.

    The tree is checked before any scoring, so a dirty tree fails in a second and not after a
    long run, and again when recording, in case HEAD moved or files changed meanwhile.
    """
    started_at = now()
    state = git_state(repo)
    if record and state.dirty:
        raise DirtyTreeError("uncommitted changes: commit them first, or pass --no-record")
    run_id = new_run_id(state.sha, started_at)

    dev = load_labelled(raw_root, "dev", files=dict(labelled_files))
    predicted = predict(dev, run_id)
    metrics = build_metrics([s.stance for s in dev], [s.year for s in dev], predicted)
    run = record_run(conn, repo, run_id, config, metrics, started_at, state.sha) if record else None
    return metrics, run
