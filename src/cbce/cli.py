"""Command line entry point: `python -m cbce.cli ingest`."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import duckdb

from cbce.ingest import ingest
from cbce.reparse import reparse
from cbce.sources.base import Fetcher
from cbce.sources.fed.adapter import FedAdapter
from cbce.sources.fetch import http_fetch, rate_limited
from cbce.store.ddl import create_schema


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cbce")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("ingest", help="discover, fetch, parse and register Fed statements")
    run.add_argument("--first-year", type=int, default=1999)
    run.add_argument("--last-year", type=int, default=None)
    run.add_argument("--limit", type=int, default=None, help="only the newest N releases")
    run.add_argument("--interval", type=float, default=3.0, help="seconds between requests")
    run.add_argument("--refetch", action="store_true", help="re-check pages already stored")
    run.add_argument("--db", type=Path, default=Path("data/processed/cbce.duckdb"))
    run.add_argument("--raw", type=Path, default=Path("data/raw"))
    redo = commands.add_parser(
        "reparse", help="re-derive documents and sentences from stored raw files (no network)"
    )
    redo.add_argument("--db", type=Path, default=Path("data/processed/cbce.duckdb"))
    return parser


def main(argv: Sequence[str] | None = None, *, fetch: Fetcher | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reparse":
        conn = duckdb.connect(str(args.db))
        create_schema(conn)
        report = reparse(FedAdapter(), conn)
        print(
            f"documents={report.documents} reclassified={report.reclassified} "
            f"reidentified={report.reidentified}"
        )
        return 0
    fetcher = fetch or rate_limited(http_fetch, args.interval)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(args.db))
    create_schema(conn)
    adapter = FedAdapter(first_year=args.first_year, last_year=args.last_year)
    report = ingest(adapter, fetcher, conn, args.raw, refetch=args.refetch, limit=args.limit)
    print(
        f"new={report.new} unchanged={report.unchanged} changed={report.changed} "
        f"skipped={report.skipped} failed={len(report.failures)}"
    )
    for url, error in report.failures:
        print(f"  FAILED {url}: {error}")
    if report.aborted:
        print("ABORTED: too many consecutive failures; check for blocking before retrying")
    return 1 if report.aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
