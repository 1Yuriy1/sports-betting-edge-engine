"""Deterministic engine package: odds data spine, pricing, anchoring, and ledger.

Per the approved Betting Agent spec, everything that computes a number lives
here; agents and scripts only call these functions. The conspiracy model in
``scripts/hermes_conspiracy_model.py`` stays a separate, labelled narrative
layer and is never an input to staking.
"""
