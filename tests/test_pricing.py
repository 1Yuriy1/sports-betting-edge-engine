"""Unit tests for the pricing math in hermes_conspiracy_model.

Every expected value below is frozen from a live run of the current
implementation (Python 3.13, pandas 2.3.3, numpy 2.5.3, 2026-09-17) —
not hand-derived. Behavior is frozen: if the model changes, these
tests fail and the change must be a deliberate decision.
"""
from __future__ import annotations

import hermes_conspiracy_model as hcm
import numpy as np
import pytest


class TestMoneylineToImplied:
    def test_negative_odds_favorite(self):
        # -205 -> 205 / (205 + 100), as implemented
        assert hcm.moneyline_to_implied(-205) == pytest.approx(0.6721311475409836)

    def test_positive_odds_underdog(self):
        # +170 -> 100 / (170 + 100), as implemented
        assert hcm.moneyline_to_implied(170) == pytest.approx(0.37037037037037035)

    def test_plus_100_is_even(self):
        assert hcm.moneyline_to_implied(100) == pytest.approx(0.5)


class TestNoVigPair:
    def test_devigs_the_170_minus_205_market(self):
        pa, pb = hcm.no_vig_pair(170, -205)
        assert pa == pytest.approx(0.35527082119976705)
        assert pb == pytest.approx(0.644729178800233)

    def test_sums_to_one(self):
        pa, pb = hcm.no_vig_pair(170, -205)
        assert pa + pb == pytest.approx(1.0)

    def test_equal_vig_market_splits_fifty_fifty(self):
        assert hcm.no_vig_pair(-110, -110) == (pytest.approx(0.5), pytest.approx(0.5))


class TestBetProfit:
    def test_underdog_win_pays_stake_times_odds_over_100(self):
        assert hcm.bet_profit(100.0, 170, True) == pytest.approx(170.0)

    def test_favorite_win_pays_stake_times_100_over_abs_odds(self):
        assert hcm.bet_profit(100.0, -205, True) == pytest.approx(48.78048780487805)

    def test_loss_loses_the_stake_regardless_of_odds(self):
        assert hcm.bet_profit(50.0, 110, False) == pytest.approx(-50.0)


class TestStakeUnits:
    """Tier boundaries, frozen from _stake_units as implemented."""

    @pytest.mark.parametrize(
        ("edge", "recommendation", "expected"),
        [
            (0.13, "PLAY", 1.0),  # >= EDGE_BIG_PLAY (0.12)
            (0.08, "PLAY", 0.75),  # >= 0.08 boundary
            (0.05, "LEAN", 0.50),  # >= EDGE_LEAN (0.05)
            (0.02, "SMALL_LEAN", 0.25),  # >= EDGE_PASS (0.02)
            (0.01, "LEAN", 0.0),  # below EDGE_PASS
            (0.30, "PASS_EXPENSIVE_FAVORITE", 0.0),  # PASS* overrides any edge
        ],
    )
    def test_tiers(self, edge, recommendation, expected):
        assert hcm._stake_units(edge, recommendation) == pytest.approx(expected)


class TestScoringMetrics:
    probs = np.array([0.72, 0.28, 0.60])
    outcomes = np.array([1, 0, 1])

    def test_log_loss(self):
        assert hcm.log_loss(self.probs, self.outcomes) == pytest.approx(
            0.3892779192366877
        )

    def test_brier(self):
        assert hcm.brier(self.probs, self.outcomes) == pytest.approx(
            0.10560000000000003
        )

    def test_brier_of_coin_flip_on_all_outcomes_is_quarter(self):
        coin = np.full(4, 0.5)
        outs = np.array([1, 0, 1, 0])
        assert hcm.brier(coin, outs) == pytest.approx(0.25)


class TestWriteSamples:
    def test_writes_the_three_sample_files(self, tmp_path):
        files = hcm.write_samples(tmp_path)
        assert sorted(f.name for f in files) == [
            "results.csv",
            "sample_game.json",
            "teams.csv",
        ]

    def test_written_files_are_non_empty(self, tmp_path):
        for f in hcm.write_samples(tmp_path):
            assert f.stat().st_size > 0
