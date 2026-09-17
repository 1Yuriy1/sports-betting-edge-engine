"""Smoke tests for scripts/snapshot_odds.py — all offline.

Mirrors tests/test_cli.py's in-process pattern (argv patched, main() called,
return code is the contract); respx keeps the HTTP fully mocked so no run
needs an API key or network access.
"""
from __future__ import annotations

import sys
from unittest import mock

import httpx
import respx
import snapshot_odds  # ruff: first-party script, resolved via conftest sys.path

from engine.odds import DEFAULT_BASE_URL
from engine.store import OddsStore

ODDS_URL = f"{DEFAULT_BASE_URL}/sports/americanfootball_nfl/odds/"
CREDIT_HEADERS = {"x-requests-remaining": "470", "x-requests-used": "30"}

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
            }
        ],
    }
]


def run_cli(argv: list[str]) -> int:
    with mock.patch.object(sys, "argv", ["snapshot_odds.py", *argv]):
        return snapshot_odds.main()


class TestSnapshotCli:
    @respx.mock
    def test_cli_smoke_exits_zero_and_stores_rows(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ODDS_API_KEY", "test-key")
        route = respx.get(ODDS_URL).mock(
            return_value=httpx.Response(200, json=FIXTURE_EVENTS, headers=CREDIT_HEADERS)
        )
        db = tmp_path / "smoke.sqlite"

        rc = run_cli(["--sport", "americanfootball_nfl", "--db", str(db)])

        assert rc == 0
        assert route.call_count == 1
        with OddsStore(db) as store:
            assert store.row_count() == 2  # one book, two sides
        out = capsys.readouterr().out
        assert "rows=2" in out
        assert "inserted=2" in out
        assert "credits_remaining=470.0" in out

    @respx.mock
    def test_cli_skips_fresh_snapshot_without_calling_the_api(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("ODDS_API_KEY", "test-key")
        db = tmp_path / "fresh.sqlite"
        # Seed a snapshot that is seconds old: interval default 12h -> skip.
        with OddsStore(db) as store:
            from engine.odds import OddsRow, utc_now_iso

            store.insert_rows(
                [
                    OddsRow(
                        fetched_at=utc_now_iso(),
                        sport_key="americanfootball_nfl",
                        game_id="evt-1",
                        commence_time="2026-09-21T17:00:00Z",
                        bookmaker="pinnacle",
                        outcome_name="Detroit Lions",
                        price=-125.0,
                        book_updated_at=None,
                    )
                ]
            )

        rc = run_cli(["--db", str(db)])

        assert rc == 0
        assert respx.calls.call_count == 0  # no credits spent
        out = capsys.readouterr().out
        assert out.startswith("skip:")

    @respx.mock
    def test_cli_missing_api_key_exits_two_with_message(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        db = tmp_path / "nokey.sqlite"

        rc = run_cli(["--db", str(db)])

        assert rc == 2
        assert "ODDS_API_KEY" in capsys.readouterr().err

    @respx.mock
    def test_cli_persistent_api_error_exits_one(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ODDS_API_KEY", "test-key")
        monkeypatch.setattr("time.sleep", lambda _s: None)  # keep the test fast
        route = respx.get(ODDS_URL).mock(return_value=httpx.Response(500, text="boom"))
        db = tmp_path / "error.sqlite"

        rc = run_cli(["--db", str(db)])

        assert rc == 1
        assert route.call_count == 3  # default RetryPolicy exhausted
        assert "failed after 3 attempts" in capsys.readouterr().err

    def test_cli_rejects_invalid_interval_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ODDS_API_KEY", "test-key")
        monkeypatch.setenv("SNAPSHOT_INTERVAL_HOURS", "not-a-number")

        rc = run_cli(["--db", str(tmp_path / "x.sqlite")])

        assert rc == 1
