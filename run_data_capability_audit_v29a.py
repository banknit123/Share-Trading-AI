from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapability:
    name: str
    history_fit: str
    intervals: str
    richer_context: str
    credential_envs: tuple[str, ...]
    note: str


PROVIDERS = (
    ProviderCapability(
        name="DhanHQ",
        history_fit="STRONG: official docs state up to 5 years of intraday history for active instruments",
        intervals="1, 5, 15, 25, 60 minute",
        richer_context="OHLCV; OI where supported for derivatives",
        credential_envs=("DHAN_ACCESS_TOKEN", "DHAN_CLIENT_ID"),
        note="Best current fit for a 1-3 year 5-minute NSE research backfill if API access is available.",
    ),
    ProviderCapability(
        name="Zerodha Kite Connect",
        history_fit="STRONG: official docs describe archived candle data spanning several years",
        intervals="1, 3, 5, 10, 15, 30, 60 minute, day",
        richer_context="OHLCV; optional OI; live quotes/depth via separate market-data APIs",
        credential_envs=("KITE_API_KEY", "KITE_ACCESS_TOKEN"),
        note="Strong historical + eventual execution candidate; expired derivative-token handling needs care.",
    ),
    ProviderCapability(
        name="Upstox",
        history_fit="MODERATE: strong current/historical APIs, but some fine-resolution history has shorter documented retention",
        intervals="custom minute intervals in V3; historical APIs available",
        richer_context="India VIX plus broader index/global indicator support in current API ecosystem",
        credential_envs=("UPSTOX_ACCESS_TOKEN",),
        note="Useful complementary source; confirm exact history depth before making it the primary 1-3 year store.",
    ),
    ProviderCapability(
        name="Yahoo/yfinance",
        history_fit="PROTOTYPE ONLY for this project",
        intervals="5-minute currently used in research",
        richer_context="basic OHLCV",
        credential_envs=(),
        note="Keep as fallback only; do not use it as the final research-quality historical source.",
    ),
)


SIGNAL_LAYERS = (
    ("Equity intraday history", "REQUIRED", "1-3 years of 5m; ideally retain 1m too"),
    ("NIFTY 50", "REQUIRED", "proper market benchmark rather than only an internal basket proxy"),
    ("BANK NIFTY", "HIGH", "banking regime/context"),
    ("India VIX", "HIGH", "volatility regime"),
    ("Sector indices", "HIGH", "sector strength/breadth and residual returns"),
    ("Previous-day + overnight context", "HIGH", "gap and multi-day regime features"),
    ("Turnover / time-of-day relative volume", "HIGH", "liquidity and participation quality"),
    ("Futures + OI", "MEDIUM-HIGH", "institutional/derivative confirmation when history is reliable"),
    ("Options IV/OI/PCR", "MEDIUM", "only after obtaining trustworthy historical snapshots"),
    ("Bid/ask/depth/ticks", "ADVANCED", "microstructure layer; source/licensing/history must be suitable"),
)


def available(envs: tuple[str, ...]) -> tuple[bool, list[str]]:
    if not envs:
        return True, []
    missing = [name for name in envs if not os.getenv(name)]
    return not missing, missing


def main() -> None:
    print("Share-Trading-AI v29A Data Capability & Signal Source Audit")
    print("NO BROKER ORDERS ARE SENT. No live trading is enabled.\n")

    print("PROVIDER CAPABILITY")
    ready = []
    for p in PROVIDERS:
        ok, missing = available(p.credential_envs)
        status = "READY" if ok and p.credential_envs else ("FALLBACK" if not p.credential_envs else "NEEDS CREDENTIALS")
        print(f"\n{p.name}: {status}")
        print(f"  historical fit: {p.history_fit}")
        print(f"  intervals:      {p.intervals}")
        print(f"  context:        {p.richer_context}")
        print(f"  note:           {p.note}")
        if missing:
            print(f"  missing env:    {', '.join(missing)}")
        if ok and p.credential_envs:
            ready.append(p.name)

    print("\nSIGNAL LAYERS FOR V29 DATASET")
    for name, priority, reason in SIGNAL_LAYERS:
        print(f"  {priority:11} {name:34} {reason}")

    print("\nV29A RECOMMENDATION")
    if "DhanHQ" in ready:
        print("  Primary historical candidate available: DhanHQ.")
        print("  Next build: Dhan instrument-master mapper + 1-3 year NSE 5m backfill into canonical storage.")
    elif "Zerodha Kite Connect" in ready:
        print("  Primary historical candidate available: Zerodha Kite Connect.")
        print("  Next build: Kite instrument-token mapper + 1-3 year NSE 5m backfill into canonical storage.")
    elif "Upstox" in ready:
        print("  Upstox credentials detected.")
        print("  Next build: verify exact available historical depth, then instrument-key mapper/backfill.")
    else:
        print("  No broker/API historical-data credentials detected in environment variables.")
        print("  Do NOT build another predictive model yet.")
        print("  Choose/connect a proper historical source first; Yahoo remains prototype fallback only.")

    print("\nTARGET CANONICAL DATASET")
    print("  equities: broad liquid NSE universe")
    print("  history:  1-3 years")
    print("  base bar: 5 minutes (retain 1-minute source when feasible)")
    print("  context:  NIFTY 50, BANK NIFTY, India VIX, sectors, gaps, breadth, turnover")
    print("  labels:   next tradable entry -> future tradable exit")
    print("  testing:  chronological + cross-stock OOS, costs + slippage")


if __name__ == "__main__":
    main()
