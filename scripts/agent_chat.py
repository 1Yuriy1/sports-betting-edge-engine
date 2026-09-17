#!/usr/bin/env python3
"""
agent_chat.py

Chat with the Hermes agent about one spot: it reasons through the current
prices with the full deterministic engine behind it (snapshot store +
ledger) at high effort, narrates in its own voice, and always carries the
mandatory disclaimer. The model never computes a number: every figure it
cites comes from a tool result, and the conspiracy layer stays
entertainment-only.

Each line you type starts a new turn; the conversation so far travels with
the question (after the cached prefix), so earlier turns stay in context
without busting the prompt cache.

Environment:
  ANTHROPIC_API_KEY     required — Anthropic API key (never hardcoded)
  AGENT_MODEL           optional — model id (default: claude-opus-5)
  ODDS_DB_PATH          optional — snapshot/ledger db path (default ./odds.sqlite)
  HERMES_TEAMS_CSV      optional — teams CSV for the conspiracy narrative layer

Usage:
  python scripts/agent_chat.py --db ./odds.sqlite
  echo "Should I like Detroit at +190?" | python scripts/agent_chat.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.agent import (
    VALID_EFFORTS,
    MissingAgentKeyError,
    build_context,
    require_agent_key,
    run_turn,
)
from engine.store import SchemaVersionError

CHAT_EFFORT = "high"  # the plan reserves high effort for one-spot reasoning
EXIT_WORDS = {"exit", "quit"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the Hermes agent about one spot (high-effort reasoning)."
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
        "--effort",
        default=CHAT_EFFORT,
        choices=VALID_EFFORTS,
        help="Reasoning effort for chat (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_agent_key()
        context = build_context(db_path=args.db, teams_csv=args.teams)
    except MissingAgentKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, SchemaVersionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    transcript: list[str] = []
    try:
        while True:
            try:
                line = input("you> " if sys.stdin.isatty() else "").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in EXIT_WORDS:
                break
            transcript.append(f"User: {line}")
            question = "\n\n".join(transcript) + "\n\nHermes:"
            try:
                result = run_turn(
                    question,
                    effort=args.effort,
                    response_format=None,
                    context=context,
                )
            except (ValueError, SchemaVersionError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            reply = result["text"]
            transcript.append(f"Hermes: {reply}")
            print(reply)
        return 0
    except MissingAgentKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        context.close()


if __name__ == "__main__":
    sys.exit(main())
