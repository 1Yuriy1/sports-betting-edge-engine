"""Pytest bootstrap: put scripts/ and examples/ on sys.path.

Mirrors the import pattern examples/run_slate.py uses to reach the model
(ROOT = parents[1], then sys.path.insert), so tests can import both
``hermes_conspiracy_model`` and ``run_slate`` without packaging.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _sub in ("scripts", "examples"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
