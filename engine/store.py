"""SQLite snapshot store: append-only odds_snapshots, schema versioning.

Stdlib sqlite3 only. The database path comes from the ``ODDS_DB_PATH``
environment variable (default ``./odds.sqlite``, gitignored) and the schema
is versioned via ``PRAGMA user_version`` so future migrations have a handle.
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

from .odds import OddsRow

DEFAULT_DB_PATH = "./odds.sqlite"
DB_PATH_ENV = "ODDS_DB_PATH"
CURRENT_SCHEMA_VERSION = 1
DEFAULT_SNAPSHOT_INTERVAL_HOURS = 12.0

# Spec art_4Yi28Bmi, SCHEMA_V1. The bets table ships with the schema version
# so the ledger module (next subtask) lands without a migration; nothing
# writes it yet.
SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
  id INTEGER PRIMARY KEY,
  fetched_at TEXT NOT NULL,           -- ISO 8601 UTC
  sport_key TEXT NOT NULL,            -- e.g. 'americanfootball_nfl'
  game_id TEXT NOT NULL,
  commence_time TEXT NOT NULL,
  bookmaker TEXT NOT NULL,
  outcome_name TEXT NOT NULL,         -- one row per side
  price REAL NOT NULL,                -- American odds
  book_updated_at TEXT,
  UNIQUE(fetched_at, game_id, bookmaker, outcome_name)
);
CREATE TABLE IF NOT EXISTS bets (
  id INTEGER PRIMARY KEY,
  placed_at TEXT NOT NULL,
  game_id TEXT NOT NULL,
  market TEXT NOT NULL DEFAULT 'h2h',
  pick TEXT NOT NULL, odds REAL NOT NULL, stake REAL NOT NULL,
  sportsbook TEXT, closing_odds REAL,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending','won','lost','pushed')),
  profit REAL, settled_at TEXT
);
"""

_INSERT_ROW_SQL = """
INSERT OR IGNORE INTO odds_snapshots
  (fetched_at, sport_key, game_id, commence_time,
   bookmaker, outcome_name, price, book_updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


class SchemaVersionError(RuntimeError):
    """The database on disk is not compatible with this code's schema version."""


class OddsStore:
    """Append-only store for normalized odds rows.

    Idempotent per fetch: the UNIQUE(fetched_at, game_id, bookmaker,
    outcome_name) constraint plus ``INSERT OR IGNORE`` means replaying the
    same fetch appends zero rows.
    """

    def __init__(self, path: str | Path | None = None):
        resolved = (
            path if path is not None else os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
        )
        self.path = Path(resolved)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def insert_rows(self, rows: Iterable[OddsRow]) -> int:
        """Append rows; returns how many were actually inserted.

        Rows already present (same fetched_at/game/bookmaker/side) are
        skipped, so a refetch of unchanged odds appends nothing.
        """
        payload = [
            (
                row.fetched_at,
                row.sport_key,
                row.game_id,
                row.commence_time,
                row.bookmaker,
                row.outcome_name,
                row.price,
                row.book_updated_at,
            )
            for row in rows
        ]
        before = self._conn.total_changes
        self._conn.executemany(_INSERT_ROW_SQL, payload)
        self._conn.commit()
        return self._conn.total_changes - before

    def row_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]

    def latest_fetched_at(self) -> str | None:
        """Newest snapshot timestamp, or None for an empty store.

        fetched_at is always written in one ISO 8601 UTC format, so the
        lexicographic MAX is a true chronological max.
        """
        row = self._conn.execute("SELECT MAX(fetched_at) FROM odds_snapshots").fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version == CURRENT_SCHEMA_VERSION:
            return
        if version > CURRENT_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"{self.path} was written by schema version {version}, newer than"
                f" this code (v{CURRENT_SCHEMA_VERSION}); update the code or move"
                " the file aside"
            )
        if version == 0:
            has_table = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table'"
                " AND name = 'odds_snapshots'"
            ).fetchone()
            if has_table:
                # A pre-versioning database cannot be adopted safely; the
                # snapshot history simply starts over.
                raise SchemaVersionError(
                    f"{self.path} predates schema versioning; move or delete it"
                    " (snapshot history starts over)"
                )
            self._conn.executescript(SCHEMA_V1)
            self._conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
            self._conn.commit()
            return
        raise SchemaVersionError(
            f"{self.path} is schema version {version}; no migration to"
            f" v{CURRENT_SCHEMA_VERSION} exists yet"
        )


def snapshot_due(
    latest_fetched_at: str | None,
    interval_hours: float = DEFAULT_SNAPSHOT_INTERVAL_HOURS,
    *,
    now: datetime | None = None,
) -> bool:
    """Credit-aware cadence: True when a new fetch should hit the API.

    An empty store is always due; otherwise only when the newest snapshot is
    at least ``interval_hours`` old. Pass ``now`` to keep the check offline
    and deterministic in tests.
    """
    if latest_fetched_at is None:
        return True
    reference = now if now is not None else datetime.now(timezone.utc)
    latest = datetime.fromisoformat(latest_fetched_at)
    age_hours = (reference - latest).total_seconds() / 3600
    return age_hours >= interval_hours
