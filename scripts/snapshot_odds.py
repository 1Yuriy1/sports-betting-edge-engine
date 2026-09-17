#!/usr/bin/env python3
"""
snapshot_odds.py

Fetch current h2h moneylines from The Odds API and append them to the local
SQLite snapshot store. Each fetch is idempotent (replaying it appends zero
rows), and the cadence is credit-aware: when the newest snapshot is younger
than SNAPSHOT_INTERVAL_HOURS (default 12 on the free tier) the API is not
called at all.

Environment:
  ODDS_API_KEY              required — The Odds API key
  ODDS_DB_PATH              optional — snapshot db path (default ./odds.sqlite)
  SNAPSHOT_INTERVAL_HOURS   optional — minimum hours between fetches (default 12)

Usage:
  python scripts/snapshot_odds.py --sport americanfootball_nfl \
      --books pinnacle,fanduel --db ./odds.sqlite
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.odds import (
    DEFAULT_REGIONS,
    MissingAPIKeyError,
    OddsAPIError,
    OddsClient,
)
from engine.store import (
    DB_PATH_ENV,
    DEFAULT_SNAPSHOT_INTERVAL_HOURS,
    OddsStore,
    SchemaVersionError,
    snapshot_due,
)

INTERVAL_ENV = "SNAPSHOT_INTERVAL_HOURS"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append an odds snapshot to the local SQLite store."
    )
    parser.add_argument(
        "--sport",
        default="americanfootball_nfl",
        help="The Odds API sport key (default: %(default)s)",
    )
    parser.add_argument(
        "--regions",
        default=DEFAULT_REGIONS,
        help="Comma-separated region keys (default: %(default)s)",
    )
    parser.add_argument(
        "--books",
        default=None,
        help="Optional comma-separated bookmaker filter, e.g. pinnacle,fanduel",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Snapshot db path (default: ${DB_PATH_ENV} or ./odds.sqlite)",
    )
    return parser.parse_args(argv)


def _split_books(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    books = [book.strip() for book in raw.split(",") if book.strip()]
    return books or None


def _interval_from_env() -> float:
    raw = os.environ.get(INTERVAL_ENV, DEFAULT_SNAPSHOT_INTERVAL_HOURS)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{INTERVAL_ENV} must be a number of hours, got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(f"{INTERVAL_ENV} must be >= 0, got {raw!r}")
    return value


def _snapshot(
    args: argparse.Namespace,
    books: list[str] | None,
    interval: float,
    client: OddsClient,
) -> int:
    with OddsStore(args.db) as store:
        latest = store.latest_fetched_at()
        if not snapshot_due(latest, interval):
            print(
                f"skip: newest snapshot {latest} is younger than {interval:g}h;"
                " no credits spent"
            )
            return 0
        result = client.fetch_h2h(args.sport, regions=args.regions, books=books)
        inserted = store.insert_rows(result.rows)
        credits = (
            result.requests_remaining
            if result.requests_remaining is not None
            else "unknown"
        )
        print(
            f"snapshot: sport={args.sport} games={result.events}"
            f" rows={len(result.rows)} inserted={inserted}"
            f" credits_remaining={credits} db={store.path}"
        )
        return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    books = _split_books(args.books)
    try:
        interval = _interval_from_env()
        with OddsClient() as client:
            return _snapshot(args, books, interval, client)
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OddsAPIError, SchemaVersionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
