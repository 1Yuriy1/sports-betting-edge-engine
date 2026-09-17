"""Offline tests for find_edges: ranked +EV rows with fractional Kelly stakes.

Fixture arithmetic (hand-derived, det-buf at the latest snapshot):
- pinnacle +150/-200 anchors DET 0.375 / BUF 0.625 exactly.
- DET best soft price: fanduel +200 (payout 3.0 beats draftkings +195);
  implied(200) = 1/3, edge = 0.375 - 1/3 = 1/24 -> edge_pct 4.1667;
  f* = 0.375 - 0.625/2 = 0.0625, quarter-Kelly stake = 15.625.
- BUF best soft price -250: implied = 5/7 = 0.714 > 0.625, no row.
kc-lv: pinnacle -110/-110 anchors 0.5/0.5; fanduel +110 on LV gives
edge = 0.5 - 100/210 = 1/42 -> edge_pct 2.381, stake = 11.3636.
"""
from __future__ import annotations

import pytest

from engine.anchor import find_edges
from engine.kelly import kelly_stake
from engine.odds import OddsRow
from engine.store import OddsStore

SPORT = "americanfootball_nfl"
COMMENCE = "2026-09-21T17:00:00Z"
T = "2026-09-17T18:00:00Z"
DET = "Detroit Lions"
BUF = "Buffalo Bills"
KC = "Kansas City Chiefs"
LV = "Las Vegas Raiders"
DET_GAME = "det-buf-2026-09-21"
LV_GAME = "kc-lv-2026-09-21"


def _row(game_id: str, bookmaker: str, outcome_name: str, price: float):
    return OddsRow(
        fetched_at=T,
        sport_key=SPORT,
        game_id=game_id,
        commence_time=COMMENCE,
        bookmaker=bookmaker,
        outcome_name=outcome_name,
        price=float(price),
        book_updated_at=None,
    )


@pytest.fixture
def slate_store(tmp_path):
    s = OddsStore(tmp_path / "odds.sqlite")
    s.insert_rows(
        [
            # det-buf: sharp anchor DET 0.375 / BUF 0.625; soft books shade DET.
            _row(DET_GAME, "pinnacle", DET, 150),
            _row(DET_GAME, "pinnacle", BUF, -200),
            _row(DET_GAME, "fanduel", DET, 200),
            _row(DET_GAME, "fanduel", BUF, -260),
            _row(DET_GAME, "draftkings", DET, 195),
            _row(DET_GAME, "draftkings", BUF, -250),
            # kc-lv: sharp anchor 0.5/0.5; fanduel misses LV low.
            _row(LV_GAME, "pinnacle", KC, -110),
            _row(LV_GAME, "pinnacle", LV, -110),
            _row(LV_GAME, "fanduel", KC, -105),
            _row(LV_GAME, "fanduel", LV, 110),
        ]
    )
    yield s
    s.close()


class TestFindEdges:
    def test_ranked_rows_follow_the_spec_shape(self, slate_store):
        rows = find_edges(slate_store, None)
        assert [r["game_id"] for r in rows] == [DET_GAME, LV_GAME]  # edge-desc order
        top = rows[0]
        assert top["pick"] == DET
        assert top["price"] == pytest.approx(200.0)  # fanduel +200 beats dk +195
        assert top["books"] == ["fanduel"]
        assert top["sharp_prob"] == pytest.approx(0.375)
        assert top["fair_odds"] == 167  # 100 * 0.625/0.375 = 166.7
        assert top["edge_pct"] == pytest.approx(4.1666667)
        assert top["stake"] == pytest.approx(15.625)
        assert top["narrative"] is None

    def test_second_row_is_the_raiders(self, slate_store):
        rows = find_edges(slate_store, None)
        second = rows[1]
        assert second["game_id"] == LV_GAME
        assert second["pick"] == LV
        assert second["price"] == pytest.approx(110.0)
        assert second["sharp_prob"] == pytest.approx(0.5)
        assert second["edge_pct"] == pytest.approx(2.3809524)
        assert second["stake"] == pytest.approx(11.3636364)

    def test_min_edge_filters_weak_rows(self, slate_store):
        assert [r["pick"] for r in find_edges(slate_store, None, min_edge=0.03)] == [DET]
        assert find_edges(slate_store, None, min_edge=0.05) == []

    def test_full_kelly_row_is_capped_at_five_percent(self, slate_store):
        rows = find_edges(slate_store, None, bankroll=100.0, kelly_fraction=1.0)
        top = next(r for r in rows if r["pick"] == DET)
        # Raw full Kelly: 0.0625 * 1.0 * 100 = 6.25 -> capped to 5% of 100.
        assert top["stake"] == pytest.approx(5.0)

    def test_games_without_a_sharp_anchor_are_skipped(self, slate_store):
        slate_store.insert_rows(
            [
                _row("soft-only-2026-09-21", "fanduel", "Arizona Cardinals", -120),
                _row("soft-only-2026-09-21", "fanduel", "Seattle Seahawks", 100),
            ]
        )
        rows = find_edges(slate_store, None)
        assert {r["game_id"] for r in rows} == {DET_GAME, LV_GAME}

    def test_stake_matches_direct_kelly_recomputation(self, slate_store):
        for row in find_edges(slate_store, None):
            expected = kelly_stake(row["sharp_prob"], row["price"], 1000.0)
            assert row["stake"] == pytest.approx(expected)

    @pytest.mark.parametrize("bankroll", [100.0, 1000.0, 25_000.0])
    @pytest.mark.parametrize("fraction", [0.1, 0.25, 1.0])
    def test_property_every_row_stakes_at_most_five_percent(
        self, slate_store, bankroll, fraction
    ):
        rows = find_edges(
            slate_store, None, bankroll=bankroll, kelly_fraction=fraction
        )
        assert rows, "fixture must produce rows for the property to mean anything"
        for row in rows:
            assert 0.0 <= row["stake"] <= 0.05 * bankroll + 1e-9, row

    def test_empty_store_returns_no_rows(self, tmp_path):
        with OddsStore(tmp_path / "empty.sqlite") as store:
            assert find_edges(store, None) == []
