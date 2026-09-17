"""Frozen-value and property tests for fractional Kelly staking.

The headline frozen value: a 5-point edge (p = 0.55) at -110 prices to a
hand-computed quarter-Kelly stake of $13.75 on a $1,000 bankroll:
  b = 100/110, f* = 0.55 - 0.45/(100/110) = 0.055,
  stake = 0.055 * 0.25 * 1000 = 13.75.
"""
from __future__ import annotations

import itertools

import pytest

from engine.kelly import kelly_stake


class TestFrozenValues:
    def test_five_point_edge_at_minus_110_stakes_13_75(self):
        assert kelly_stake(0.55, -110, 1000.0) == pytest.approx(13.75)

    def test_five_point_edge_at_plus_110(self):
        # f* = 0.55 - 0.45/1.1 = 0.1409090..., quarter Kelly on 1000.
        assert kelly_stake(0.55, 110, 1000.0) == pytest.approx(35.22727272727273)

    def test_half_fraction_doubles_the_stake(self):
        assert kelly_stake(0.55, -110, 1000.0, fraction=0.5) == pytest.approx(27.5)

    def test_stake_scales_linearly_with_bankroll_below_the_cap(self):
        assert kelly_stake(0.55, -110, 2000.0) == pytest.approx(27.5)


class TestPassWhenEdgeNonPositive:
    @pytest.mark.parametrize(
        ("prob", "odds"),
        [
            (0.45, -110),  # f* = 0.45 - 0.55/(100/110) < 0
            (0.30, -200),  # heavy juice against a thin win probability
            (0.50, 100),  # exactly breakeven: f* = 0.5 - 0.5/1.0 = 0
        ],
    )
    def test_non_positive_edge_always_stakes_zero(self, prob, odds):
        assert kelly_stake(prob, odds, 1000.0) == 0.0


class TestFivePercentCap:
    def test_ninety_percent_at_plus_110_caps_at_five_percent(self):
        # Raw quarter Kelly: 0.809091 * 0.25 * 1000 = 202.27 -> capped at 50.
        assert kelly_stake(0.90, 110, 1000.0) == pytest.approx(50.0)

    def test_full_kelly_still_capped(self):
        assert kelly_stake(0.90, 110, 1000.0, fraction=1.0) == pytest.approx(50.0)

    def test_property_every_stake_within_zero_to_five_percent(self):
        combos = itertools.product(
            (0.51, 0.6, 0.75, 0.9, 0.99),
            (-400, -200, -110, 100, 150, 1000),
            (500.0, 10_000.0),
            (0.25, 1.0),
        )
        for prob, odds, bankroll, fraction in combos:
            stake = kelly_stake(prob, odds, bankroll, fraction=fraction)
            assert 0.0 <= stake <= 0.05 * bankroll + 1e-9, (
                prob,
                odds,
                bankroll,
                fraction,
            )


class TestValidation:
    def test_zero_odds_rejected(self):
        with pytest.raises(ValueError, match="not a price"):
            kelly_stake(0.55, 0, 1000.0)

    def test_probability_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="within"):
            kelly_stake(1.2, -110, 1000.0)
        with pytest.raises(ValueError, match="within"):
            kelly_stake(-0.1, -110, 1000.0)

    def test_nonpositive_bankroll_rejected(self):
        with pytest.raises(ValueError, match="bankroll"):
            kelly_stake(0.55, -110, 0.0)

    def test_nonpositive_fraction_rejected(self):
        with pytest.raises(ValueError, match="fraction"):
            kelly_stake(0.55, -110, 1000.0, fraction=0.0)
