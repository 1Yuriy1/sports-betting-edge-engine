"""Fractional Kelly staking: quarter Kelly by default, capped at 5% bankroll.

Product stakes always come from here — never from the conspiracy model's
0.25–1.00 unit ladder, which stays in the narrative layer only.
"""
from __future__ import annotations

DEFAULT_FRACTION = 0.25
KELLY_CAP_FRACTION = 0.05


def kelly_stake(
    prob: float,
    odds: float,
    bankroll: float,
    fraction: float = DEFAULT_FRACTION,
) -> float:
    """Fractional Kelly stake in dollars.

    f* = p - (1-p)/b where b is the net payout per unit staked
    (odds/100 for positive American prices, 100/|odds| for negative);
    stake = f* * fraction * bankroll, clamped to [0, 0.05 * bankroll].
    Returns 0.0 whenever the edge is non-positive — a pass.
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be within [0, 1], got {prob}")
    if odds == 0.0:
        raise ValueError(f"American odds of {odds} are not a price")
    if bankroll <= 0.0:
        raise ValueError(f"bankroll must be positive, got {bankroll}")
    if fraction <= 0.0:
        raise ValueError(f"Kelly fraction must be positive, got {fraction}")
    b = odds / 100.0 if odds > 0.0 else 100.0 / abs(odds)
    f_star = prob - (1.0 - prob) / b
    if f_star <= 0.0:
        return 0.0
    cap = KELLY_CAP_FRACTION * bankroll
    return float(min(max(f_star * fraction * bankroll, 0.0), cap))
