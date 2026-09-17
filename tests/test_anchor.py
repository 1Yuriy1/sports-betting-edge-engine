"""Offline tests for the sharp anchor, line movement, and pricing re-exports.

Every expected probability is hand-derived from no_vig_pair's definition
(divide each side's implied probability by the book's overround). The
canonical fixture pair: +150/-200 devigs to (0.375, 0.625) —
implied(+150) = 0.4, implied(-200) = 2/3, overround = 16/15,
0.4 / (16/15) = 3/8 exactly. Fixtures write OddsRow batches straight
into a temp-dir OddsStore — no API, no key.
"""
from __future__ import annotations

import hermes_conspiracy_model as hcm
import pytest

from engine.anchor import AnchorUnavailable, line_move, sharp_anchor
from engine.odds import OddsRow
from engine.pricing import moneyline_to_implied, no_vig_pair
from engine.store import OddsStore

SPORT = "americanfootball_nfl"
COMMENCE = "2026-09-21T17:00:00Z"
GAME = "det-buf-2026-09-21"
T1 = "2026-09-17T12:00:00Z"
T2 = "2026-09-17T18:00:00Z"
DET = "Detroit Lions"
BUF = "Buffalo Bills"


def _row(game_id: str, fetched_at: str, bookmaker: str, outcome_name: str, price: float):
    return OddsRow(
        fetched_at=fetched_at,
        sport_key=SPORT,
        game_id=game_id,
        commence_time=COMMENCE,
        bookmaker=bookmaker,
        outcome_name=outcome_name,
        price=float(price),
        book_updated_at=None,
    )


@pytest.fixture
def store(tmp_path):
    s = OddsStore(tmp_path / "odds.sqlite")
    yield s
    s.close()


class TestPricingReexports:
    def test_no_vig_pair_is_the_frozen_function(self):
        assert no_vig_pair is hcm.no_vig_pair

    def test_moneyline_to_implied_is_the_frozen_function(self):
        assert moneyline_to_implied is hcm.moneyline_to_implied

    def test_reexport_devigs_the_canonical_pair(self):
        assert no_vig_pair(150, -200) == (pytest.approx(0.375), pytest.approx(0.625))


class TestSharpAnchor:
    def test_single_sharp_book_devigs_the_known_pair(self, store):
        store.insert_rows(
            [
                _row(GAME, T1, "pinnacle", DET, 150),
                _row(GAME, T1, "pinnacle", BUF, -200),
            ]
        )
        anchor = sharp_anchor(store, GAME)
        assert anchor["market"] == "h2h"
        assert anchor["books"] == ("pinnacle",)
        assert anchor["probs"][DET] == pytest.approx(0.375)
        assert anchor["probs"][BUF] == pytest.approx(0.625)

    def test_two_sharp_books_are_averaged(self, store):
        store.insert_rows(
            [
                _row(GAME, T1, "pinnacle", DET, 150),
                _row(GAME, T1, "pinnacle", BUF, -200),
                _row(GAME, T1, "circa", DET, -110),
                _row(GAME, T1, "circa", BUF, -110),
            ]
        )
        anchor = sharp_anchor(store, GAME)
        assert anchor["books"] == ("pinnacle", "circa")
        assert anchor["probs"][DET] == pytest.approx(0.4375)  # (0.375 + 0.5) / 2
        assert anchor["probs"][BUF] == pytest.approx(0.5625)

    def test_one_sided_sharp_book_is_skipped_not_substituted(self, store):
        store.insert_rows(
            [
                _row(GAME, T1, "pinnacle", DET, 150),
                _row(GAME, T1, "pinnacle", BUF, -200),
                _row(GAME, T1, "circa", DET, -110),  # circa quotes one side only
            ]
        )
        anchor = sharp_anchor(store, GAME)
        assert anchor["books"] == ("pinnacle",)
        assert anchor["probs"][DET] == pytest.approx(0.375)
        assert anchor["probs"][BUF] == pytest.approx(0.625)

    def test_soft_books_alone_never_anchor(self, store):
        store.insert_rows(
            [
                _row(GAME, T1, "fanduel", DET, 130),
                _row(GAME, T1, "fanduel", BUF, -150),
                _row(GAME, T1, "draftkings", DET, 125),
                _row(GAME, T1, "draftkings", BUF, -145),
            ]
        )
        with pytest.raises(AnchorUnavailable, match="never"):
            sharp_anchor(store, GAME)

    def test_incomplete_sharp_book_does_not_fall_back_to_soft_books(self, store):
        store.insert_rows(
            [
                _row(GAME, T1, "pinnacle", DET, 150),  # pinnacle misses BUF
                _row(GAME, T1, "fanduel", DET, 130),
                _row(GAME, T1, "fanduel", BUF, -150),
            ]
        )
        with pytest.raises(AnchorUnavailable):
            sharp_anchor(store, GAME)

    def test_unknown_game_raises(self, store):
        with pytest.raises(AnchorUnavailable, match="no stored snapshots"):
            sharp_anchor(store, "no-such-game")

    def test_anchor_uses_the_latest_snapshot_not_a_stale_one(self, store):
        store.insert_rows(
            [
                _row(GAME, T1, "pinnacle", DET, 150),
                _row(GAME, T1, "pinnacle", BUF, -200),
                _row(GAME, T2, "pinnacle", DET, -110),
                _row(GAME, T2, "pinnacle", BUF, -110),
            ]
        )
        anchor = sharp_anchor(store, GAME)
        assert anchor["probs"][DET] == pytest.approx(0.5)
        assert anchor["probs"][BUF] == pytest.approx(0.5)

    def test_non_h2h_market_is_rejected(self, store):
        with pytest.raises(ValueError, match="h2h"):
            sharp_anchor(store, GAME, market="spreads")

    def test_three_way_book_cannot_form_a_two_sided_pair(self, store):
        # Soccer-style h2h has a third outcome; no_vig_pair is two-sided,
        # so such books are skipped and coverage can end up unavailable.
        store.insert_rows(
            [
                _row("epl-2026-09-21", T1, "pinnacle", "Arsenal", -150),
                _row("epl-2026-09-21", T1, "pinnacle", "Chelsea", 400),
                _row("epl-2026-09-21", T1, "pinnacle", "Draw", 260),
            ]
        )
        with pytest.raises(AnchorUnavailable):
            sharp_anchor(store, "epl-2026-09-21")


class TestLineMove:
    @pytest.fixture
    def moved(self, store):
        store.insert_rows(
            [
                _row(GAME, T1, "pinnacle", DET, 150),
                _row(GAME, T1, "pinnacle", BUF, -200),
                _row(GAME, T1, "fanduel", DET, 160),
                _row(GAME, T1, "fanduel", BUF, -230),
                _row(GAME, T2, "pinnacle", DET, 140),
                _row(GAME, T2, "pinnacle", BUF, -210),
                _row(GAME, T2, "fanduel", DET, 150),
                _row(GAME, T2, "fanduel", BUF, -240),
                _row(GAME, T2, "draftkings", DET, 145),  # late-joining book
                _row(GAME, T2, "draftkings", BUF, -225),
            ]
        )
        return store

    def test_single_book_open_to_now(self, moved):
        move = line_move(moved, GAME, book="pinnacle")
        assert move[DET]["open"] == pytest.approx(150.0)
        assert move[DET]["now"] == pytest.approx(140.0)
        assert move[DET]["delta"] == pytest.approx(-10.0)
        assert move[BUF]["delta"] == pytest.approx(-10.0)

    def test_bookless_averages_every_book_at_each_endpoint(self, moved):
        move = line_move(moved, GAME)
        assert move[DET]["open"] == pytest.approx(155.0)  # (150 + 160) / 2
        assert move[DET]["now"] == pytest.approx(145.0)  # (140 + 150 + 145) / 3
        assert move[BUF]["open"] == pytest.approx(-215.0)
        assert move[BUF]["now"] == pytest.approx(-225.0)

    def test_late_joining_book_has_no_open(self, moved):
        move = line_move(moved, GAME, book="draftkings")
        assert move[DET]["open"] is None
        assert move[DET]["now"] == pytest.approx(145.0)
        assert move[DET]["delta"] is None

    def test_single_snapshot_is_zero_movement(self, store):
        store.insert_rows(
            [
                _row(GAME, T1, "pinnacle", DET, 150),
                _row(GAME, T1, "pinnacle", BUF, -200),
            ]
        )
        move = line_move(store, GAME, book="pinnacle")
        assert move[DET]["open"] == pytest.approx(150.0)
        assert move[DET]["delta"] == pytest.approx(0.0)

    def test_unknown_game_returns_empty(self, moved):
        assert line_move(moved, "no-such-game") == {}


class TestStoreSnapshotsReader:
    """The read accessor the anchor relies on: filter + ordering."""

    def test_snapshots_filters_by_game_and_orders_by_fetch(self, store):
        store.insert_rows(
            [
                _row("g2", T2, "pinnacle", DET, 140),
                _row("g1", T1, "pinnacle", DET, 150),
                _row("g1", T2, "circa", DET, -110),
            ]
        )
        rows = store.snapshots(game_id="g1")
        assert [(r.fetched_at, r.bookmaker) for r in rows] == [
            (T1, "pinnacle"),
            (T2, "circa"),
        ]

    def test_snapshots_without_filter_returns_everything(self, store):
        store.insert_rows(
            [
                _row("g1", T1, "pinnacle", DET, 150),
                _row("g2", T2, "circa", DET, -110),
            ]
        )
        assert len(store.snapshots()) == 2
