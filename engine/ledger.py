"""Bet ledger: log, grade, and closing-line-value reporting over the bets table.

Manual grading for v1 — the caller supplies ``(game_id, outcome)`` pairs and
profit is computed with the frozen model's ``bet_profit`` semantics
(``scripts/hermes_conspiracy_model.py``), reimplemented here so the ledger
stays stdlib-only (the frozen module pulls in pandas).

CLV (closing line value) compares the price you took with the closing price
logged on the bet, in decimal-odds terms:

    clv_pct = (decimal_odds_taken / decimal_odds_close - 1) * 100

Positive means you beat the close. Bets without logged closing odds cannot
carry CLV and are counted in the report, not averaged.
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

from .odds import utc_now_iso
from .store import (
    CURRENT_SCHEMA_VERSION,
    DB_PATH_ENV,
    DEFAULT_DB_PATH,
    SCHEMA_V1,
    SchemaVersionError,
)

GRADED_OUTCOMES = ("won", "lost", "pushed")


class GradeResult(TypedDict):
    game_id: str
    outcome: str
    graded: int


class ClvBet(TypedDict):
    id: int
    game_id: str
    pick: str
    odds: float
    closing_odds: float
    status: str
    clv_pct: float


class ClvReport(TypedDict):
    mean_clv_pct: float | None
    bets_graded: int
    bets_ungraded: int
    bets_missing_closing_odds: int
    per_bet: list[ClvBet]


def bet_profit(stake: float, odds: float, outcome: str) -> float:
    """Net profit for a settled bet; mirrors the frozen model's bet_profit.

    Won at +150 returns ``stake * 150 / 100``; won at -150 returns
    ``stake * 100 / 150``; a loss costs the stake; a push returns nothing
    (the frozen module takes a ``won`` bool, so the push case is explicit
    here).
    """
    if outcome == "pushed":
        return 0.0
    if outcome == "lost":
        return -stake
    if odds > 0:
        return stake * odds / 100.0
    return stake * 100.0 / abs(odds)


def american_to_decimal(odds: float) -> float:
    """American price to decimal odds (total return per 1 staked)."""
    odds = float(odds)
    if odds == 0:
        raise ValueError(f"odds must be nonzero American odds, got {odds}")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def bet_clv_pct(odds: float, closing_odds: float) -> float:
    """Percent CLV of a bet taken at ``odds`` against its closing price.

    Positive when you beat the close: took +120, closed +100 is +10.0.
    """
    return (american_to_decimal(odds) / american_to_decimal(closing_odds) - 1.0) * 100.0


def open_ledger(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) a database with the bets table.

    Mirrors OddsStore's path resolution and schema-version gate so the CLI
    and the snapshot store agree on which file is current. SCHEMA_V1 is
    idempotent (CREATE TABLE IF NOT EXISTS), so sharing the DDL never
    rewrites an existing table.
    """
    resolved = (
        path if path is not None else os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
    )
    db_path = Path(resolved)
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > CURRENT_SCHEMA_VERSION:
        conn.close()
        raise SchemaVersionError(
            f"{db_path} was written by schema version {version}, newer than"
            f" this code (v{CURRENT_SCHEMA_VERSION}); update the code or move"
            " the file aside"
        )
    if version == 0 and conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'odds_snapshots'"
    ).fetchone():
        conn.close()
        raise SchemaVersionError(
            f"{db_path} predates schema versioning; move or delete it"
        )
    conn.executescript(SCHEMA_V1)
    if version == 0:
        conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
    conn.commit()
    return conn


def log_bet(
    conn: sqlite3.Connection,
    *,
    game_id: str,
    pick: str,
    odds: float,
    stake: float,
    market: str = "h2h",
    sportsbook: str | None = None,
    closing_odds: float | None = None,
    placed_at: str | None = None,
) -> int:
    """Insert a pending bet; returns its row id.

    ``odds`` is the American price taken. ``closing_odds`` is the same-side
    price at close when it is known at logging time — grading never sets it.
    """
    odds = float(odds)
    stake = float(stake)
    if odds == 0:
        raise ValueError(f"odds must be nonzero American odds, got {odds}")
    if stake <= 0:
        raise ValueError(f"stake must be positive, got {stake}")
    with conn:
        cur = conn.execute(
            "INSERT INTO bets (placed_at, game_id, market, pick, odds, stake,"
            " sportsbook, closing_odds, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (
                placed_at if placed_at is not None else utc_now_iso(),
                game_id,
                market,
                pick,
                odds,
                stake,
                sportsbook,
                closing_odds,
            ),
        )
    return int(cur.lastrowid)


def grade_bets(
    conn: sqlite3.Connection,
    grades: Iterable[tuple[str, str]],
) -> list[GradeResult]:
    """Settle pending bets from ``(game_id, outcome)`` pairs (manual, v1).

    Every pending bet on the game gets the outcome, profit is computed via
    ``bet_profit``, and ``settled_at`` is stamped now. The batch is atomic:
    an unknown outcome or a game with no pending bets raises ValueError and
    rolls the whole batch back.
    """
    pairs = list(grades)
    for _, outcome in pairs:
        if outcome not in GRADED_OUTCOMES:
            raise ValueError(
                f"outcome must be one of {GRADED_OUTCOMES}, got {outcome!r}"
            )
    results: list[GradeResult] = []
    settled_at = utc_now_iso()
    with conn:
        for game_id, outcome in pairs:
            rows = conn.execute(
                "SELECT id, odds, stake FROM bets"
                " WHERE game_id = ? AND status = 'pending'",
                (game_id,),
            ).fetchall()
            if not rows:
                raise ValueError(f"no pending bet found for game_id={game_id!r}")
            for row in rows:
                profit = bet_profit(row["stake"], row["odds"], outcome)
                conn.execute(
                    "UPDATE bets SET status = ?, profit = ?, settled_at = ?"
                    " WHERE id = ?",
                    (outcome, profit, settled_at, row["id"]),
                )
            results.append(
                {
                    "game_id": game_id,
                    "outcome": outcome,
                    "graded": len(rows),
                }
            )
    return results


def clv_report(conn: sqlite3.Connection) -> ClvReport:
    """CLV over the ledger: per-bet and mean, off logged closing odds.

    Ungraded (pending) bets are excluded from the aggregates and reported
    via ``bets_ungraded``; graded bets without closing odds land in
    ``bets_missing_closing_odds``. The mean is computed from the rounded
    per-bet values so it matches the report as printed.
    """
    rows = conn.execute(
        "SELECT id, game_id, pick, odds, closing_odds, status FROM bets"
        " ORDER BY id"
    ).fetchall()
    per_bet: list[ClvBet] = []
    ungraded = 0
    missing = 0
    for row in rows:
        if row["status"] == "pending":
            ungraded += 1
            continue
        if row["closing_odds"] is None or float(row["closing_odds"]) == 0.0:
            missing += 1
            continue
        per_bet.append(
            {
                "id": row["id"],
                "game_id": row["game_id"],
                "pick": row["pick"],
                "odds": row["odds"],
                "closing_odds": row["closing_odds"],
                "status": row["status"],
                "clv_pct": round(bet_clv_pct(row["odds"], row["closing_odds"]), 2),
            }
        )
    mean = (
        round(sum(bet["clv_pct"] for bet in per_bet) / len(per_bet), 2)
        if per_bet
        else None
    )
    return {
        "mean_clv_pct": mean,
        "bets_graded": len(per_bet) + missing,
        "bets_ungraded": ungraded,
        "bets_missing_closing_odds": missing,
        "per_bet": per_bet,
    }
