"""Compatibility runner for v27A.

The original v27A expects BROAD_UNIVERSE from v26B, while v26B exposes the
same broad liquid-NSE list as RESEARCH_UNIVERSE.  This runner aliases the
current name before importing v27A, without changing any research logic.
"""
from __future__ import annotations

import run_market_general_edge_v26b as v26b

if not hasattr(v26b, "BROAD_UNIVERSE"):
    v26b.BROAD_UNIVERSE = v26b.RESEARCH_UNIVERSE

import run_broad_feature_diagnostic_v27a as v27a


if __name__ == "__main__":
    print("v27A compatibility fix: using v26B RESEARCH_UNIVERSE as BROAD_UNIVERSE")
    v27a.main()
