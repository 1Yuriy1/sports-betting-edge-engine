#!/usr/bin/env python3
"""
bet_log.py

Log, grade, and report bets in the local SQLite ledger (the ``bets`` table
that ships with schema v1). Grading is manual for v1: you supply the game
and the outcome, and profit is computed with the frozen model's bet_profit
semantics. The report prints mean and per-bet CLV off the closing odds
logged on each bet; pending bets are excluded from the mean and counted.

Environment:
  ODDS_DB_PATH   optional — ledger db path (default ./odds.sqlite)

Usage:
  python scripts/bet_log.py log --game-id nfl-w2-det-buf --pick DET \
      --odds -150 --stake 100 --sportsbook pinnacle --closing-odds -160
  python scripts/bet_log.py grade --game-id nfl-w2-det-buf --outcome won
  python scripts/bet_log.py report
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.ledger import (
    GRADED_OUTCOMES,
    SchemaVersionError,
    clv_report,
    grade_bets,
    log_bet,
    open_ledger,
)
from engine.store import DB_PATH_ENV


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log, grade, and report bets in the SQLite ledger."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--db",
        default=None,
        help=f"Ledger db path (default: ${DB_PATH_ENV} or ./odds.sqlite)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    log_parser = sub.add_parser("log", parents=[common], help="Log a pending bet.")
    log_parser.add_argument("--game-id", required=True, help="Game identifier")
    log_parser.add_argument("--pick", required=True, help="Side taken, e.g. DET")
    log_parser.add_argument(
        "--odds", type=float, required=True, help="American price taken"
    )
    log_parser.add_argument("--stake", type=float, required=True)
    log_parser.add_argument("--market", default="h2h", help="(default: %(default)s)")
    log_parser.add_argument("--sportsbook", default=None)
    log_parser.add_argument(
        "--closing-odds",
        type=float,
        default=None,
        help="Optional American price logged at close (enables CLV)",
    )

    grade_parser = sub.add_parser(
        "grade", parents=[common], help="Settle a game's pending bets (manual)."
    )
    grade_parser.add_argument("--game-id", required=True)
    grade_parser.add_argument("--outcome", required=True, choices=GRADED_OUTCOMES)

    sub.add_parser("report", parents=[common], help="Print the CLV report as JSON.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        conn = open_ledger(args.db)
        try:
            if args.command == "log":
                bet_id = log_bet(
                    conn,
                    game_id=args.game_id,
                    pick=args.pick,
                    odds=args.odds,
                    stake=args.stake,
                    market=args.market,
                    sportsbook=args.sportsbook,
                    closing_odds=args.closing_odds,
                )
                closing = (
                    f"{args.closing_odds:g}"
                    if args.closing_odds is not None
                    else "n/a"
                )
                print(
                    f"logged bet id={bet_id} game={args.game_id} pick={args.pick}"
                    f" odds={args.odds:g} stake={args.stake:g} closing_odds={closing}"
                )
            elif args.command == "grade":
                (graded,) = grade_bets(conn, [(args.game_id, args.outcome)])
                print(
                    f"graded {graded['graded']} bet(s) for {graded['game_id']}:"
                    f" {graded['outcome']}"
                )
            else:
                print(json.dumps(clv_report(conn), indent=2))
        finally:
            conn.close()
        return 0
    except (ValueError, SchemaVersionError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
