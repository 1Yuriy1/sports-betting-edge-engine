#!/usr/bin/env python3
"""Run the V2 conspiracy model over a CSV slate.

Usage:
  uv run --with pandas --with numpy examples/run_slate.py \
    --teams examples/sample_teams.csv \
    --slate examples/week2_fox_odds.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hermes_conspiracy_model import ConspiracyModel


def val(row, key, default=0):
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    try:
        return float(raw)
    except ValueError:
        return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", required=True)
    ap.add_argument("--slate", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model = ConspiracyModel.from_csv(args.teams)
    rows = []
    with open(args.slate, newline="") as f:
        for row in csv.DictReader(f):
            a, b = row["team_a"].upper(), row["team_b"].upper()
            ctx = {"a": {}, "b": {}, "shared": {}}
            for k, v in row.items():
                if k.startswith("ctx_a_") and v not in ("", None):
                    ctx["a"][k[6:]] = float(v)
                elif k.startswith("ctx_b_") and v not in ("", None):
                    ctx["b"][k[6:]] = float(v)
                elif k.startswith("ctx_") and v not in ("", None):
                    ctx["shared"][k[4:]] = float(v)
            out = model.market_edge(
                a,
                b,
                float(row["ml_a"]),
                float(row["ml_b"]),
                ctx,
                spread=float(row["spread"]) if row.get("spread") else None,
                week=int(float(row.get("week") or 1)),
            )
            v2 = out["v2_edge"]
            rows.append({
                "game": f"{a} @ {b}",
                "pick": v2["model_pick"],
                "model_prob": round(v2["model_pick_prob"] * 100, 1),
                "market_prob": round(v2["market_pick_prob"] * 100, 1),
                "edge_pct": v2["edge_pct"],
                "recommendation": v2["recommendation"],
                "stake_units": v2["stake_units"],
                "trap_flags": ",".join(v2["trap_flags"]) or "-",
            })

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print("| Game | Pick | Model % | Market % | Edge | V2 | Units | Traps |")
        print("|---|---:|---:|---:|---:|---|---:|---|")
        for r in rows:
            print(
                f"| {r['game']} | {r['pick']} | {r['model_prob']}% | "
                f"{r['market_prob']}% | {r['edge_pct']} | {r['recommendation']} | "
                f"{r['stake_units']} | {r['trap_flags']} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
