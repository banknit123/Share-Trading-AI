from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ai.data.market_data import download_history
from run_candle_dominant_v12 import _to_utc
from run_market_general_edge_v26b import RESEARCH_UNIVERSE

ROOT = Path("data/v29")
HISTORY_DIR = ROOT / "history"
MANIFEST_PATH = ROOT / "v29a2_history_manifest.csv"
SUMMARY_PATH = ROOT / "v29a2_history_summary.json"

INTERVAL = os.getenv("V29_INTERVAL", "5m")
YAHOO_PERIOD = os.getenv("V29_YAHOO_PERIOD", "60d")
PROVIDER_REQUEST = os.getenv("V29_PROVIDER", "auto").strip().lower()

# Context instruments are deliberately separated from the equity research universe.
# Yahoo names are used only for prototype plumbing. Broker/provider adapters can map
# these canonical names to their own security IDs later.
CONTEXT_YAHOO = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
}


@dataclass
class BackfillResult:
    canonical_symbol: str
    provider_symbol: str
    provider: str
    asset_type: str
    status: str
    rows: int = 0
    first_ts: str = ""
    last_ts: str = ""
    sessions: int = 0
    duplicate_rows: int = 0
    nonpositive_prices: int = 0
    missing_ohlcv_cells: int = 0
    path: str = ""
    note: str = ""


def choose_provider() -> str:
    if PROVIDER_REQUEST not in {"auto", "yahoo", "dhan"}:
        raise ValueError("V29_PROVIDER must be auto, yahoo, or dhan")
    if PROVIDER_REQUEST == "yahoo":
        return "yahoo"
    if PROVIDER_REQUEST == "dhan":
        return "dhan"
    if os.getenv("DHAN_CLIENT_ID") and os.getenv("DHAN_ACCESS_TOKEN"):
        return "dhan"
    return "yahoo"


def provider_ready(provider: str) -> tuple[bool, str]:
    if provider == "yahoo":
        return True, "prototype fallback only"
    missing = [k for k in ("DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN") if not os.getenv(k)]
    if missing:
        return False, "missing environment variables: " + ", ".join(missing)
    # A Dhan historical request also needs exchange segment + security ID mapping.
    # We deliberately do not invent IDs. v29A3 will load the official instrument master.
    mapping = ROOT / "dhan_security_map.csv"
    if not mapping.exists():
        return False, f"credentials found but {mapping} is not built yet; run the Dhan instrument-master step first"
    return True, "credentials and security map detected"


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    x = _to_utc(df).copy()
    rename = {c: c.capitalize() for c in x.columns}
    x = x.rename(columns=rename)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in x.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    x = x[needed].copy()
    x.index.name = "timestamp"
    return x.sort_index()


def quality(df: pd.DataFrame) -> dict[str, int]:
    px = df[["Open", "High", "Low", "Close"]]
    idx = pd.DatetimeIndex(df.index)
    ist = idx.tz_convert("Asia/Kolkata")
    return {
        "rows": int(len(df)),
        "sessions": int(pd.Series(ist.date).nunique()),
        "duplicate_rows": int(idx.duplicated().sum()),
        "nonpositive_prices": int((px <= 0).any(axis=1).sum()),
        "missing_ohlcv_cells": int(df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum()),
    }


def save_symbol(df: pd.DataFrame, provider: str, canonical_symbol: str) -> Path:
    folder = HISTORY_DIR / provider / INTERVAL
    folder.mkdir(parents=True, exist_ok=True)
    safe = canonical_symbol.replace("^", "INDEX_").replace("/", "_")
    path = folder / f"{safe}.csv.gz"
    out = df.reset_index()
    out.to_csv(path, index=False, compression="gzip")
    return path


def yahoo_download(provider_symbol: str) -> pd.DataFrame:
    raw = download_history(provider_symbol, period=YAHOO_PERIOD, interval=INTERVAL)
    if raw is None or raw.empty:
        raise RuntimeError("no market data returned")
    return standardize(raw)


def dhan_download(canonical_symbol: str, provider_symbol: str) -> pd.DataFrame:
    # v29A2 intentionally stops before an unsafe partial Dhan implementation.
    # Dhan uses security IDs and exchange segments rather than Yahoo tickers.
    # Once registration is complete, v29A3 will load Dhan's official instrument
    # master and implement chunked multi-year historical requests into this same store.
    raise RuntimeError(
        "Dhan adapter not activated yet. Credentials/security IDs must be resolved via the official instrument master in v29A3."
    )


def backfill_one(canonical: str, provider_symbol: str, provider: str, asset_type: str) -> BackfillResult:
    try:
        if provider == "yahoo":
            df = yahoo_download(provider_symbol)
        else:
            df = dhan_download(canonical, provider_symbol)
        q = quality(df)
        path = save_symbol(df, provider, canonical)
        return BackfillResult(
            canonical_symbol=canonical,
            provider_symbol=provider_symbol,
            provider=provider,
            asset_type=asset_type,
            status="OK",
            rows=q["rows"],
            first_ts=pd.Timestamp(df.index.min()).isoformat(),
            last_ts=pd.Timestamp(df.index.max()).isoformat(),
            sessions=q["sessions"],
            duplicate_rows=q["duplicate_rows"],
            nonpositive_prices=q["nonpositive_prices"],
            missing_ohlcv_cells=q["missing_ohlcv_cells"],
            path=str(path),
        )
    except Exception as exc:
        return BackfillResult(
            canonical_symbol=canonical,
            provider_symbol=provider_symbol,
            provider=provider,
            asset_type=asset_type,
            status="SKIP",
            note=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    provider = choose_provider()
    ready, provider_note = provider_ready(provider)

    print("Share-Trading-AI v29A2 Historical Backfill Framework")
    print("NO BROKER ORDERS ARE SENT. Data engineering only.")
    print(f"Requested provider: {PROVIDER_REQUEST} | selected: {provider}")
    print(f"Provider status: {'READY' if ready else 'NOT READY'} - {provider_note}")
    print(f"Interval: {INTERVAL}")

    if provider == "dhan" and not ready:
        print("\nDhan is not ready, so no fake/made-up security mappings will be used.")
        print("Set V29_PROVIDER=yahoo if you want to test the prototype storage pipeline now.")
        return

    items: list[tuple[str, str, str]] = []
    if provider == "yahoo":
        items.extend((s, s, "EQUITY") for s in RESEARCH_UNIVERSE)
        items.extend((canonical, ticker, "CONTEXT") for canonical, ticker in CONTEXT_YAHOO.items())
    else:
        # Mapping-driven Dhan population will be activated by v29A3.
        mapping = pd.read_csv(ROOT / "dhan_security_map.csv")
        for r in mapping.itertuples(index=False):
            items.append((str(r.canonical_symbol), str(r.security_id), str(r.asset_type)))

    results: list[BackfillResult] = []
    for i, (canonical, provider_symbol, asset_type) in enumerate(items, start=1):
        print(f"{i:3d}/{len(items):3d} {canonical:18} ... ", end="", flush=True)
        r = backfill_one(canonical, provider_symbol, provider, asset_type)
        results.append(r)
        if r.status == "OK":
            print(f"OK rows={r.rows:,} sessions={r.sessions} {r.first_ts[:10]}->{r.last_ts[:10]}")
        else:
            print(f"SKIP {r.note}")

    manifest = pd.DataFrame([r.__dict__ for r in results])
    manifest.to_csv(MANIFEST_PATH, index=False)

    ok = manifest[manifest["status"] == "OK"].copy()
    summary = {
        "provider": provider,
        "provider_note": provider_note,
        "interval": INTERVAL,
        "requested_items": int(len(manifest)),
        "successful_items": int(len(ok)),
        "successful_equities": int((ok["asset_type"] == "EQUITY").sum()) if not ok.empty else 0,
        "successful_context": int((ok["asset_type"] == "CONTEXT").sum()) if not ok.empty else 0,
        "total_rows": int(ok["rows"].sum()) if not ok.empty else 0,
        "median_sessions": float(ok["sessions"].median()) if not ok.empty else 0.0,
        "research_grade": bool(provider != "yahoo"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nV29A2 BACKFILL SUMMARY")
    print(f"  provider: {summary['provider']}")
    print(f"  successful items: {summary['successful_items']}/{summary['requested_items']}")
    print(f"  equities stored: {summary['successful_equities']}")
    print(f"  context instruments stored: {summary['successful_context']}")
    print(f"  total stored rows: {summary['total_rows']:,}")
    print(f"  median sessions/item: {summary['median_sessions']:.0f}")
    print(f"  research-grade history: {'YES' if summary['research_grade'] else 'NO - prototype only'}")
    print("\nFILES")
    print(f"  {MANIFEST_PATH}")
    print(f"  {SUMMARY_PATH}")
    print(f"  {HISTORY_DIR / provider / INTERVAL}")
    print("\nNEXT")
    if provider == "yahoo":
        print("  Storage/quality pipeline is testable now, but do not train the final v29B model from this 60-day fallback.")
        print("  When Dhan registration is complete, v29A3 will build the official security-ID map and multi-year backfill into the same schema.")
    else:
        print("  Dhan history is present; next build the canonical multi-source feature dataset before v29B.")


if __name__ == "__main__":
    main()
