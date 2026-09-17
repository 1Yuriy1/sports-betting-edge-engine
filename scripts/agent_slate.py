#!/usr/bin/env python3
"""
agent_slate.py

Run one Hermes agent turn over the current slate and print the structured
report. The agent routes through the deterministic engine tools (snapshot
store + ledger), narrates the result, and always carries the mandatory
disclaimer. The model never computes a number: every figure in the report
is copied from a tool result.

Environment:
  ANTHROPIC_API_KEY     required — Anthropic API key (never hardcoded)
  AGENT_MODEL           optional — model id (default: claude-opus-5)
  ODDS_DB_PATH          optional — snapshot/ledger db path (default ./odds.sqlite)
  HERMES_TEAMS_CSV      optional — teams CSV for the conspiracy narrative layer
                        (see scripts/snapshot_odds.py for the data spine)

Usage:
  python scripts/agent_slate.py --db ./odds.sqlite --effort medium
  python scripts/agent_slate.py --question "Any edges on today's NFL slate?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.agent import (
    DEFAULT_EFFORT,
    VALID_EFFORTS,
    MissingAgentKeyError,
    build_context,
    require_agent_key,
    run_turn,
)
from engine.store import SchemaVersionError

DEFAULT_QUESTION = (
    "Build today's slate report: list the games, then the current +EV edges"
    " with stakes, and narrate the conspiracy segment."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One Hermes agent turn over the current slate (structured report)."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Snapshot/ledger db path (default: $ODDS_DB_PATH or ./odds.sqlite)",
    )
    parser.add_argument(
        "--teams",
        default=None,
        help="Teams CSV for the narrative layer (default: $HERMES_TEAMS_CSV or examples/sample_teams.csv)",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="What to ask the agent (default: the standard slate report ask)",
    )
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=VALID_EFFORTS,
        help="Reasoning effort for routine reports (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_agent_key()
        with build_context(db_path=args.db, teams_csv=args.teams) as context:
            result = run_turn(
                args.question,
                effort=args.effort,
                context=context,
            )
        print(json.dumps(result["structured"], indent=2))
        return 0
    except MissingAgentKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, SchemaVersionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
