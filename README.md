# cbce: Central Bank Communication Engine

Turns Federal Reserve text into hawkish / neutral / dovish stance scores, then asks whether
those scores explain same-day moves in short-dated government bond yields.

The project is judged on method, not on a trading edge. A well-evidenced null result is a
valid outcome, and nothing here is investment advice.

**Status:** milestone 1 of 3 is complete. The Fed statement archive (1999 to date) is ingested,
and five baseline scorers are evaluated on a held-out development split. The LLM extractor,
the yield event study and the public site are not built yet.

## Why the evaluation is built the way it is

A stance classifier is easy to over-claim. Three decisions guard against that.

**Whole-year splits.** The labelled sentences come from the "Trillion Dollar Words" dataset.
Its own train/test split is random per sentence, so sentences from the same meeting land on both
sides. This repo pools the data and splits by calendar year instead. Test years (1998, 2002,
2006, 2010, 2014, 2018, 2022) are frozen in code and refused by the loader unless a run ID is
supplied. All tuning uses the other 20 years.

**Uncertainty that respects clustering.** Sentences within a year are not independent, so
confidence intervals come from a bootstrap that resamples whole years, not sentences.
Differences between scorers are computed on the same resamples, which gives a paired interval
that says whether one scorer really beats another.

**Every number has a run ID.** `make eval-dev` refuses to run on a dirty git tree. Each result
is stored with the commit SHA and a hash of its full configuration, so a figure can always be
traced back to the code and settings that produced it.

## Current results

Development split, 1,917 sentences across 20 years, macro-F1 with a 95% year-clustered
bootstrap interval. Recorded as `run-20261029T234511Z-5510603`.

| Scorer | Macro-F1 | 95% CI |
| --- | --- | --- |
| Fine-tuned DistilRoBERTa (5-fold, grouped by year) | 0.520 | 0.493 to 0.543 |
| FinBERT (inference only) | 0.486 | 0.462 to 0.510 |
| Hand-written word list | 0.475 | 0.456 to 0.494 |
| Loughran-McDonald tone | 0.407 | 0.390 to 0.424 |
| Majority class | 0.219 | 0.210 to 0.226 |

The encoder beats both dictionaries, but its lead over FinBERT (+0.034, interval -0.001 to
+0.064) is not distinguishable from zero. FinBERT and the hand-written list are also tied. A
short word list matching a pretrained model is the main thing these baselines tell us, and it
sets the bar the LLM extractor has to clear.

## Pipeline

```
discover -> fetch -> parse -> segment -> register -> score -> evaluate -> publish
```

Each ingest step is idempotent and keyed by content hash. Raw pages are stored once, by hash,
and never edited. Parsing and sentence segmentation can be re-run from the raw files with no
network access (`make reparse`).

| Stage | Where | Notes |
| --- | --- | --- |
| Sources | `src/cbce/sources/` | One adapter per bank behind a shared interface. Fed only so far. |
| Parse | `src/cbce/parse/` | Rule-based sentence splitter with a versioned id scheme. |
| Store | `src/cbce/store/` | Content-addressed raw files, DuckDB tables, a `runs` ledger. |
| Score | `src/cbce/scoring/` | All scorers share one interface and one record type. |
| Evaluate | `src/cbce/eval/` | Splits, metrics, bootstrap intervals, cross-validation. |

Every release carries a `published_at` timestamp taken from the release itself, never from
fetch time, so later stages can enforce point-in-time correctness. Releases before 2016 only
state "for immediate release", so those timestamps are recorded at day precision and flagged.

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
make setup      # create the environment
make test       # offline test suite
make lint       # ruff
```

Fetching from federalreserve.gov identifies the caller in the User-Agent. Set a contact
address in your environment first; it is read at run time and never stored in the repo.

```sh
export CBCE_CONTACT="you@example.com"
make ingest ARGS="--first-year 2021 --limit 10"
```

The labelled data and the Loughran-McDonald dictionary are not included, for licensing reasons
(see below). Place them in `data/raw/` to reproduce the baseline table with `make eval-dev`.
Add `ARGS="--no-record --no-encoder"` for a quick run that skips the encoder and writes nothing.
The first full run trains the encoder, which takes about 25 minutes on a laptop CPU; later runs
reuse the cached predictions.

## Known limits

- The labelled sentences come from minutes, press conferences and speeches. None are
  post-meeting statements, so sentence-level accuracy says nothing direct about statement text.
- They were drawn after a keyword filter, so they are not a random sample of all sentences.
- The dataset has been public since 2023, so a hosted language model may have seen it.
- Only daily yields are available, so the statement, the press conference and the rest of the
  day cannot be separated. No fed funds futures are used, so what the market had already priced
  is partly absorbed into the estimates.
- About eight meetings a year is a small sample. Intervals are reported as they come.
- The encoder was trained with a single seed.

## Data and licences

- Labelled sentences: Shah, Paturi and Chava, "Trillion Dollar Words", ACL 2023. CC BY-NC 4.0.
  Used for non-commercial research only, with attribution, and never redistributed here.
- Loughran-McDonald master dictionary: free for academic research, cite the authors. Not
  redistributed here.
- FinBERT (`ProsusAI/finbert`, CC BY-NC-SA 4.0): used for inference only, pinned to a fixed
  revision, never fine-tuned or redistributed.
- Encoder base model: `distilbert/distilroberta-base`, Apache-2.0, pinned to a fixed revision.
- Federal Reserve statements are public releases from federalreserve.gov.

Because of the non-commercial terms above, the published site must never be monetised.
