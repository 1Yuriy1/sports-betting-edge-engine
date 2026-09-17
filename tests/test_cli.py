"""CLI smoke tests for the three invocations documented in the README.

Invocations 1 and 2 run the documented command line verbatim in a
subprocess (return code is part of the contract). Invocation 3 calls
``run_slate.main()`` in-process with argv patched.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import run_slate

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "scripts" / "hermes_conspiracy_model.py"
README_MATCHUP_ARGS = [
    "--teams",
    "examples/sample_teams.csv",
    "--matchup",
    "DET",
    "BUF",
    "--ml-a",
    "170",
    "--ml-b",
    "-205",
    "--spread",
    "4.5",
    "--week",
    "2",
    "--json",
]
README_SLATE_ARGS = [
    "--teams",
    "examples/sample_teams.csv",
    "--slate",
    "examples/week2_fox_odds.csv",
]


class TestMatchupJson:
    def test_documented_invocation_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(MODEL), *README_MATCHUP_ARGS],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,  # rc is the assertion
        )
        assert result.returncode == 0, result.stderr

    def test_v2_edge_exposes_documented_keys(self):
        result = subprocess.run(
            [sys.executable, str(MODEL), *README_MATCHUP_ARGS],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,  # rc is the assertion
        )
        payload = json.loads(result.stdout)
        v2 = payload["v2_edge"]
        for key in ("model_pick", "recommendation", "edge_pct"):
            assert key in v2
        # Frozen from live runs, 2026-09-17 (behavior-freeze check).
        assert v2["model_pick"] == "BUF"
        assert v2["recommendation"] == "PASS_NO_EDGE"
        assert v2["edge_pct"] == 0.67


class TestWriteSamples:
    def test_documented_invocation_writes_three_files(self, tmp_path):
        dest = tmp_path / "generated"
        result = subprocess.run(
            [sys.executable, str(MODEL), "--write-samples", str(dest)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,  # rc is the assertion
        )
        assert result.returncode == 0, result.stderr
        assert sorted(p.name for p in dest.iterdir()) == [
            "results.csv",
            "sample_game.json",
            "teams.csv",
        ]
        for p in dest.iterdir():
            assert p.stat().st_size > 0


class TestRunSlate:
    def test_slate_over_example_csvs_prints_markdown_picks_table(self, capsys):
        argv = ["run_slate.py", *README_SLATE_ARGS]
        with mock.patch.object(sys, "argv", argv):
            rc = run_slate.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert out.startswith("| Game | Pick | Model % | Market % | Edge | V2 | Units | Traps |")
        assert "| DET @ BUF | BUF |" in out
        assert out.count("\n") > 2  # header, separator, and data rows
