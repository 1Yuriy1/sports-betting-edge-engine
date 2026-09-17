"""Offline tests for the bet ledger: grading, CLV math, and the CLI.

Fixtures are seeded straight into a tmp_path SQLite file — no API, no key,
no network. All fixtures live in this file so tasks that run in parallel
can own their own test files without sharing conftest state.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from engine.ledger import (
    bet_clv_pct,
    bet_profit,
    clv_report,
    grade_bets,
    log_bet,
    open_ledger,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BET_LOG = REPO_ROOT / "scripts" / "bet_log.py"

# (game_id, pick, odds, stake, outcome, expected profit to the cent).
# Expected values come from the frozen model's bet_profit semantics:
# won at +105 x 100 stake = 105.00; won at -150 x 100 = 66.67;
# lost = -stake; pushed = 0.00.
KNOWN_BETS = [
    ("nfl-w2-det-buf", "DET", -150.0, 100.0, "won", 66.67),
    ("nfl-w2-car-atl", "ATL", 105.0, 100.0, "won", 105.00),
    ("nfl-w2-gb-chi", "GB", -120.0, 50.0, "lost", -50.00),
    ("nfl-w2-kc-cin", "KC", -110.0, 110.0, "lost", -110.00),
    ("nfl-w2-sea-sf", "SF", 150.0, 40.0, "won", 60.00),
    ("nfl-w2-dal-nyg", "DAL", -105.0, 100.0, "pushed", 0.00),
    ("nfl-w2-phi-kc", "PHI", -200.0, 100.0, "won", 50.00),
    ("nfl-w2-nyj-pit", "NYJ", 250.0, 20.0, "lost", -20.00),
    ("nfl-w2-mia-ne", "MIA", -135.0, 67.5, "won", 50.00),
    ("nfl-w2-chi-hou", "CHI", 180.0, 25.0, "won", 45.00),
]


@pytest.fixture
def ledger(tmp_path):
    return open_ledger(tmp_path / "bets.sqlite")


class TestPureMath:
    def test_bet_profit_matches_frozen_semantics(self):
        assert bet_profit(100, 150, "won") == 150.0
        assert bet_profit(100, -150, "won") == pytest.approx(66.666666)
        assert bet_profit(50, -120, "lost") == -50.0
        assert bet_profit(100, -110, "pushed") == 0.0

    def test_bet_clv_sign_convention(self):
        # Took +120, closed +100: the line shortened toward your pick → positive.
        assert bet_clv_pct(120, 100) == pytest.approx(10.0)
        # Took -110, closed -120: the line shortened toward your pick too —
        # you locked the better price, so you beat the close → positive.
        assert bet_clv_pct(-110, -120) == pytest.approx(4.132231405)


class TestLogBet:
    def test_log_bet_inserts_pending_row(self, ledger):
        bet_id = log_bet(ledger, game_id="g1", pick="DET", odds=-150, stake=100)
        row = ledger.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
        assert row["game_id"] == "g1"
        assert row["pick"] == "DET"
        assert row["market"] == "h2h"
        assert row["status"] == "pending"
        assert row["profit"] is None
        assert row["closing_odds"] is None
        assert row["placed_at"].endswith("+00:00")

    def test_log_bet_rejects_zero_odds_and_nonpositive_stake(self, ledger):
        with pytest.raises(ValueError, match="odds"):
            log_bet(ledger, game_id="g1", pick="DET", odds=0, stake=100)
        with pytest.raises(ValueError, match="stake"):
            log_bet(ledger, game_id="g1", pick="DET", odds=-150, stake=0)


class TestGrading:
    def test_ten_known_bets_grade_to_exact_profit(self, ledger):
        for game_id, pick, odds, stake, _, _ in KNOWN_BETS:
            log_bet(ledger, game_id=game_id, pick=pick, odds=odds, stake=stake)
        pairs = [(game_id, outcome) for game_id, *_z, outcome, _p in KNOWN_BETS]
        results = grade_bets(ledger, pairs)
        assert [r["graded"] for r in results] == [1] * len(KNOWN_BETS)
        rows = ledger.execute(
            "SELECT game_id, status, profit, settled_at FROM bets ORDER BY id"
        ).fetchall()
        for (game_id, _pick, _odds, _stake, outcome, expected), row in zip(
            KNOWN_BETS, rows
        ):
            assert row["game_id"] == game_id
            assert row["status"] == outcome
            assert round(row["profit"], 2) == expected  # exact to the cent
            assert row["settled_at"]

    def test_grade_unknown_game_raises(self, ledger):
        with pytest.raises(ValueError, match="no pending bet"):
            grade_bets(ledger, [("missing-game", "won")])

    def test_grade_invalid_outcome_raises_and_rolls_back(self, ledger):
        log_bet(ledger, game_id="g1", pick="DET", odds=-150, stake=100)
        with pytest.raises(ValueError, match="outcome"):
            grade_bets(ledger, [("g1", "forfeit")])
        row = ledger.execute("SELECT status FROM bets").fetchone()
        assert row["status"] == "pending"


class TestClvReport:
    def test_mean_clv_matches_hand_computation(self, ledger):
        # Hand arithmetic in decimal odds (dec = 1 + 100/|am| for favorites).
        # CLV = dec_taken / dec_close - 1; positive = beat the close:
        #   bet A: took -110 (dec 1.909091), closed -120 (dec 1.833333):
        #          1.909091 / 1.833333 - 1 = +4.13%
        #   bet B: took +120 (dec 2.20), closed +150 (dec 2.50):
        #          2.20 / 2.50 - 1 = -12.00%
        #   bet C: took -105, closed -105 → 0.00%
        #   mean of rounded values = (4.13 - 12.00 + 0.00) / 3 = -2.62%
        log_bet(
            ledger, game_id="A", pick="KC", odds=-110, stake=100, closing_odds=-120
        )
        log_bet(
            ledger, game_id="B", pick="SF", odds=120, stake=100, closing_odds=150
        )
        log_bet(
            ledger, game_id="C", pick="DET", odds=-105, stake=100, closing_odds=-105
        )
        grade_bets(ledger, [("A", "won"), ("B", "lost"), ("C", "pushed")])
        report = clv_report(ledger)
        assert [b["clv_pct"] for b in report["per_bet"]] == [4.13, -12.0, 0.0]
        assert report["mean_clv_pct"] == -2.62
        assert report["bets_ungraded"] == 0

    def test_ungraded_bets_excluded_and_counted(self, ledger):
        # Hand arithmetic, CLV = dec_taken / dec_close - 1:
        #   A: took -110 (dec 1.909091), closed -125 (dec 1.80):
        #      1.909091 / 1.80 - 1 = +6.06%
        #   B: took +120 (dec 2.20), closed +110 (dec 2.10):
        #      2.20 / 2.10 - 1 = +4.76%
        #   mean of rounded values = (6.06 + 4.76) / 2 = 5.41%.
        log_bet(
            ledger, game_id="A", pick="KC", odds=-110, stake=100, closing_odds=-125
        )
        log_bet(
            ledger, game_id="B", pick="SF", odds=120, stake=100, closing_odds=110
        )
        log_bet(
            ledger,
            game_id="PENDING",
            pick="DET",
            odds=-150,
            stake=100,
            closing_odds=-160,
        )
        grade_bets(ledger, [("A", "won"), ("B", "lost")])
        report = clv_report(ledger)
        assert report["bets_ungraded"] == 1
        assert report["bets_graded"] == 2
        assert [b["game_id"] for b in report["per_bet"]] == ["A", "B"]
        assert report["mean_clv_pct"] == 5.41

    def test_graded_without_closing_odds_counted_not_averaged(self, ledger):
        log_bet(ledger, game_id="A", pick="KC", odds=-110, stake=100)
        log_bet(
            ledger, game_id="B", pick="SF", odds=120, stake=100, closing_odds=110
        )
        grade_bets(ledger, [("A", "won"), ("B", "lost")])
        report = clv_report(ledger)
        assert report["bets_missing_closing_odds"] == 1
        assert report["bets_graded"] == 2  # A is graded, just not in the mean
        assert [b["game_id"] for b in report["per_bet"]] == ["B"]
        # B: took +120 (dec 2.20), closed +110 (dec 2.10) → +4.76%.
        assert report["mean_clv_pct"] == 4.76

    def test_empty_ledger_report(self, ledger):
        report = clv_report(ledger)
        assert report["mean_clv_pct"] is None
        assert report["per_bet"] == []
        assert report["bets_graded"] == 0
        assert report["bets_ungraded"] == 0
        assert report["bets_missing_closing_odds"] == 0


class TestSchemaConstraints:
    def test_check_constraint_rejects_invalid_status(self, ledger):
        with pytest.raises(sqlite3.IntegrityError):
            ledger.execute(
                "INSERT INTO bets (placed_at, game_id, pick, odds, stake, status)"
                " VALUES ('2026-09-17T19:00:00+00:00', 'g1', 'DET', -150, 100,"
                " 'void')"
            )


class TestBetLogCli:
    @staticmethod
    def _run(tmp_path, *argv):
        env = {**os.environ, "ODDS_DB_PATH": str(tmp_path / "bets.sqlite")}
        return subprocess.run(
            [sys.executable, str(BET_LOG), *argv],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
            check=False,  # rc is the assertion
        )

    def test_documented_flow_exits_zero(self, tmp_path):
        # Mirrors tests/test_cli.py: run the README command line in a
        # subprocess; the return code is part of the contract.
        result = self._run(
            tmp_path,
            "log",
            "--game-id",
            "nfl-w2-det-buf",
            "--pick",
            "DET",
            "--odds",
            "-150",
            "--stake",
            "100",
            "--closing-odds",
            "-160",
        )
        assert result.returncode == 0, result.stderr
        result = self._run(
            tmp_path,
            "log",
            "--game-id",
            "nfl-w2-car-atl",
            "--pick",
            "ATL",
            "--odds",
            "105",
            "--stake",
            "50",
        )
        assert result.returncode == 0, result.stderr
        result = self._run(
            tmp_path, "grade", "--game-id", "nfl-w2-det-buf", "--outcome", "won"
        )
        assert result.returncode == 0, result.stderr
        result = self._run(tmp_path, "report")
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        # -150 closed at -160: 1.666667 / 1.625 - 1 = +2.56%.
        assert report["mean_clv_pct"] == pytest.approx(2.56)
        assert report["bets_ungraded"] == 1  # the ungraded ATL bet is counted
        assert [b["game_id"] for b in report["per_bet"]] == ["nfl-w2-det-buf"]

    def test_invalid_outcome_exits_nonzero(self, tmp_path):
        result = self._run(
            tmp_path, "grade", "--game-id", "g", "--outcome", "void"
        )
        assert result.returncode != 0
        assert "error" in result.stderr.lower()
