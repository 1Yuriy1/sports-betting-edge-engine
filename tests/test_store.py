"""Offline tests for the SQLite snapshot store: idempotency and versioning.

Everything runs against a temp-directory SQLite file; no API, no key.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from engine.odds import DEFAULT_BASE_URL, OddsClient, OddsRow, utc_now_iso
from engine.store import (
    CURRENT_SCHEMA_VERSION,
    DB_PATH_ENV,
    DEFAULT_SNAPSHOT_INTERVAL_HOURS,
    OddsStore,
    SchemaVersionError,
    snapshot_due,
)

ODDS_URL = f"{DEFAULT_BASE_URL}/sports/americanfootball_nfl/odds/"
CREDIT_HEADERS = {"x-requests-remaining": "480", "x-requests-used": "20"}

FIXTURE_EVENTS = [
    {
        "id": "evt-1",
        "sport_key": "americanfootball_nfl",
        "commence_time": "2026-09-21T17:00:00Z",
        "home_team": "Detroit Lions",
        "away_team": "Buffalo Bills",
        "bookmakers": [
            {
                "key": "pinnacle",
                "last_update": "2026-09-17T18:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Detroit Lions", "price": -125},
                            {"name": "Buffalo Bills", "price": 105},
                        ],
                    }
                ],
            },
            {
                "key": "fanduel",
                "last_update": "2026-09-17T17:55:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Detroit Lions", "price": -120},
                            {"name": "Buffalo Bills", "price": 100},
                        ],
                    }
                ],
            },
        ],
    }
]


def make_row(fetched_at: str = "2026-09-17T19:00:00+00:00", **overrides) -> OddsRow:
    values = {
        "fetched_at": fetched_at,
        "sport_key": "americanfootball_nfl",
        "game_id": "evt-1",
        "commence_time": "2026-09-21T17:00:00Z",
        "bookmaker": "pinnacle",
        "outcome_name": "Detroit Lions",
        "price": -125.0,
        "book_updated_at": "2026-09-17T18:00:00Z",
    }
    values.update(overrides)
    return OddsRow(**values)


class TestInsertAndIdempotency:
    def test_fetch_parse_store_roundtrip(self, tmp_path):
        with respx.mock:
            respx.get(ODDS_URL).mock(
                return_value=httpx.Response(
                    200, json=FIXTURE_EVENTS, headers=CREDIT_HEADERS
                )
            )
            client = OddsClient(api_key="test-key", sleep_fn=lambda _s: None)
            result = client.fetch_h2h("americanfootball_nfl")
            client.close()

        with OddsStore(tmp_path / "odds.sqlite") as store:
            assert store.insert_rows(result.rows) == 4
            assert store.row_count() == 4
            row = store._conn.execute(
                "SELECT * FROM odds_snapshots WHERE bookmaker = 'pinnacle'"
                " AND outcome_name = 'Buffalo Bills'"
            ).fetchone()
            assert row["game_id"] == "evt-1"
            assert row["price"] == 105.0
            assert row["sport_key"] == "americanfootball_nfl"
            assert row["book_updated_at"] == "2026-09-17T18:00:00Z"

    def test_second_identical_run_appends_zero_duplicate_rows(self, tmp_path):
        with OddsStore(tmp_path / "odds.sqlite") as store:
            rows = [make_row(), make_row(bookmaker="fanduel", price=-120.0)]
            assert store.insert_rows(rows) == 2
            assert store.insert_rows(rows) == 0  # identical refetch: no-op
            assert store.row_count() == 2

    def test_new_fetch_with_new_timestamp_appends(self, tmp_path):
        with OddsStore(tmp_path / "odds.sqlite") as store:
            assert store.insert_rows([make_row()]) == 1
            newer = make_row(fetched_at="2026-09-17T20:00:00+00:00", price=-115.0)
            assert store.insert_rows([newer]) == 1  # history accumulates
            assert store.row_count() == 2


class TestSchemaVersioning:
    def test_fresh_db_is_stamped_with_current_version(self, tmp_path):
        with OddsStore(tmp_path / "odds.sqlite") as store:
            version = store._conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == CURRENT_SCHEMA_VERSION == 1

    def test_bets_table_ships_in_schema_v1(self, tmp_path):
        with OddsStore(tmp_path / "odds.sqlite") as store:
            tables = {
                row[0]
                for row in store._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            assert {"odds_snapshots", "bets"} <= tables

    def test_reopening_existing_db_is_a_noop(self, tmp_path):
        path = tmp_path / "odds.sqlite"
        with OddsStore(path) as store:
            store.insert_rows([make_row()])
        with OddsStore(path) as store:
            assert store.row_count() == 1

    def test_newer_schema_version_is_refused(self, tmp_path):
        path = tmp_path / "odds.sqlite"
        with OddsStore(path):
            pass
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        conn.close()
        with pytest.raises(SchemaVersionError, match="newer than"):
            OddsStore(path)

    def test_preversioning_db_is_refused(self, tmp_path):
        path = tmp_path / "odds.sqlite"
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE odds_snapshots (id INTEGER PRIMARY KEY, fetched_at TEXT)"
        )
        conn.commit()
        conn.close()  # tables exist, user_version still 0
        with pytest.raises(SchemaVersionError, match="predates schema versioning"):
            OddsStore(path)


class TestDbPathResolution:
    def test_env_var_selects_the_db_path(self, tmp_path, monkeypatch):
        db = tmp_path / "nested" / "env.sqlite"
        monkeypatch.setenv(DB_PATH_ENV, str(db))
        with OddsStore() as store:
            assert store.path == db
            assert store.row_count() == 0
        assert db.exists()

    def test_default_path_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv(DB_PATH_ENV, raising=False)
        monkeypatch.chdir(tmp_path)
        with OddsStore() as store:
            assert store.path.name == "odds.sqlite"
        assert (tmp_path / "odds.sqlite").exists()

    def test_explicit_path_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DB_PATH_ENV, str(tmp_path / "env.sqlite"))
        with OddsStore(tmp_path / "explicit.sqlite") as store:
            assert store.path.name == "explicit.sqlite"


class TestSnapshotCadence:
    def test_empty_store_is_always_due(self):
        assert snapshot_due(None) is True

    def test_fresh_snapshot_skips_the_api_call(self):
        now = datetime.now(timezone.utc)
        one_hour_ago = (now - timedelta(hours=1)).isoformat()
        assert snapshot_due(one_hour_ago, 12.0, now=now) is False

    def test_stale_snapshot_is_due(self):
        now = datetime.now(timezone.utc)
        thirteen_hours_ago = (now - timedelta(hours=13)).isoformat()
        assert snapshot_due(thirteen_hours_ago, 12.0, now=now) is True

    def test_boundary_age_is_due(self):
        now = datetime.now(timezone.utc)
        exactly_interval = (now - timedelta(hours=12)).isoformat()
        assert snapshot_due(exactly_interval, DEFAULT_SNAPSHOT_INTERVAL_HOURS, now=now)

    def test_utc_now_iso_is_parseable_and_offset_marked(self):
        parsed = datetime.fromisoformat(utc_now_iso())
        assert parsed.tzinfo is not None
