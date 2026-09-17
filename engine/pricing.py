"""No-vig pricing primitives, re-exported from the frozen conspiracy model.

``scripts/hermes_conspiracy_model.py`` is behaviour-frozen (its tests are
part of the spec), so the engine imports its pricing math instead of
copying it — one source of truth for ``no_vig_pair`` and
``moneyline_to_implied``. The script is a module, not a package, so its
directory joins ``sys.path`` here if it is not already present (the test
bootstrap and examples do the same for their imports).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from hermes_conspiracy_model import moneyline_to_implied, no_vig_pair

__all__ = ["moneyline_to_implied", "no_vig_pair"]
