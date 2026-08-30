from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_intraday_portfolio_v21 import IST, order_costs

ROOT = Path("data/v29")
PROVIDER = os.getenv("V29_PROVIDER", "yahoo").strip().lower()
INTERVAL = os.getenv("V29_INTERVAL", "5m")
HISTORY_DIR = ROOT / "history" / PROVIDER / INTERVAL
MANIFEST_PATH = ROOT / "v29a2_history_manifest.csv"
OUT_DIR = ROOT / "canonical"

HORIZONS = {"5m": 1, "10m": 2, "30m": 6, "60m": 12, "120m": 24}
NOTIONAL = 300000.0
SLIPPAGE_BPS_PER_SIDE = 2.0
MIN_STOCKS_PER_TIMESTAMP = 20

CONTEXT = {"NIFTY50", "BANKNIFTY", "INDIAVIX"}


def _read_history(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, parse_dates=["timestamp"])
    ts = pd.DatetimeIndex(x["timestamp"])
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    x["timestamp"] = ts
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _safe_file(symbol: str) -> Path:
    safe = symbol.replace("^", "INDEX_").replace("/", "_")
    return HISTORY_DIR / f"{safe}.csv.gz"


def _round_trip_net(gross: float) -> float:
    sell_value = max(0.0, NOTIONAL * (1.0 + gross))
    _, buy_cost = order_costs(NOTIONAL, "BUY")
    _, sell_cost = order_costs(sell_value, "SELL")
    slip = 2.0 * SLIPPAGE_BPS_PER_SIDE / 10000.0
    return gross - (buy_cost + sell_cost) / NOTIONAL - slip


def _basic_features(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy().sort_values("timestamp")
    close = x["Close"]
    vol = x["Volume"]
    for n in [1, 3, 6, 12, 24]:
        x[f"ret_{n}"] = close.pct_change(n)
    x["ema_6"] = close.ewm(span=6, adjust=False).mean()
    x["ema_18"] = close.ewm(span=18, adjust=False).mean()
    x["ema_36"] = close.ewm(span=36, adjust=False).mean()
    x["ema_spread_6_18"] = x["ema_6"] / x["ema_18"] - 1.0
    x["ema_spread_18_36"] = x["ema_18"] / x["ema_36"] - 1.0
    x["volatility_12"] = x["ret_1"].rolling(12).std()
    x["volatility_36"] = x["ret_1"].rolling(36).std()
    x["volume_ratio_12"] = vol / vol.rolling(12).mean().replace(0, np.nan)

    ist = x["timestamp"].dt.tz_convert(IST)
    x["date"] = ist.dt.strftime("%Y-%m-%d")
    x["clock"] = ist.dt.strftime("%H:%M")
    x["minute_of_day"] = ist.dt.hour * 60 + ist.dt.minute
    x["session"] = np.select(
        [
            (x["clock"] >= "09:15") & (x["clock"] < "11:15"),
            (x["clock"] >= "11:15") & (x["clock"] < "13:15"),
            (x["clock"] >= "13:15") & (x["clock"] <= "15:30"),
        ],
        ["OPEN", "MID", "LATE"],
        default="OTHER",
    )
    return x


def _add_previous_day_context(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy().sort_values("timestamp")
    daily = x.groupby("date").agg(
        day_open=("Open", "first"),
        day_high=("High", "max"),
        day_low=("Low", "min"),
        day_close=("Close", "last"),
        day_volume=("Volume", "sum"),
    )
    daily["prev_close"] = daily["day_close"].shift(1)
    daily["prev_high"] = daily["day_high"].shift(1)
    daily["prev_low"] = daily["day_low"].shift(1)
    daily["prev_day_return"] = daily["day_close"].shift(1) / daily["day_open"].shift(1) - 1.0
    daily["prev_day_range"] = daily["day_high"].shift(1) / daily["day_low"].shift(1) - 1.0
    daily = daily[["prev_close", "prev_high", "prev_low", "prev_day_return", "prev_day_range"]].reset_index()
    x = x.merge(daily, on="date", how="left")
    x["overnight_gap"] = x.groupby("date")["Open"].transform("first") / x["prev_close"] - 1.0
    x["from_day_open"] = x["Close"] / x.groupby("date")["Open"].transform("first") - 1.0
    return x


def _past_only_tod_relative_volume(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy().sort_values("timestamp")
    # Compare each bar with prior days at the same clock time. shift() prevents the current day
    # and all future days from entering the baseline.
    grp = x.groupby("clock", sort=False)["Volume"]
    x["tod_volume_baseline"] = grp.transform(lambda s: s.shift(1).expanding(min_periods=5).median())
    x["tod_relative_volume"] = x["Volume"] / x["tod_volume_baseline"].replace(0, np.nan)
    grp_ret = x.groupby("clock", sort=False)["volatility_12"]
    x["tod_volatility_baseline"] = grp_ret.transform(lambda s: s.shift(1).expanding(min_periods=5).median())
    x["tod_relative_volatility"] = x["volatility_12"] / x["tod_volatility_baseline"].replace(0, np.nan)
    return x


def _context_frame(symbol: str, prefix: str) -> pd.DataFrame:
    p = _safe_file(symbol)
    if not p.exists():
        raise FileNotFoundError(f"Missing required context history: {p}")
    x = _basic_features(_read_history(p))
    keep = ["timestamp", "Close", "ret_1", "ret_3", "ret_6", "ret_12", "volatility_12"]
    x = x[keep].copy()
    return x.rename(columns={c: f"{prefix}_{c.lower()}" for c in keep if c != "timestamp"})


def build_equity_panel() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Run v29A2 first; missing {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)
    eq = manifest[(manifest["status"] == "OK") & (manifest["asset_type"] == "EQUITY")].copy()
    frames = []
    for i, r in enumerate(eq.itertuples(index=False), start=1):
        symbol = str(r.canonical_symbol)
        p = Path(str(r.path)) if pd.notna(r.path) and str(r.path) else _safe_file(symbol)
        if not p.exists():
            p = _safe_file(symbol)
        try:
            x = _read_history(p)
            x = _basic_features(x)
            x = _add_previous_day_context(x)
            x = _past_only_tod_relative_volume(x)
            x["symbol"] = symbol
            frames.append(x)
            print(f"{i:2d}/{len(eq):2d} {symbol:15} rows={len(x):,}")
        except Exception as exc:
            print(f"{i:2d}/{len(eq):2d} {symbol:15} SKIP {type(exc).__name__}: {exc}")
    if not frames:
        raise RuntimeError("No equity history could be loaded")
    panel = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    panel = panel[panel["session"] != "OTHER"].copy()
    return panel


def add_market_context(panel: pd.DataFrame) -> pd.DataFrame:
    n50 = _context_frame("NIFTY50", "nifty")
    bank = _context_frame("BANKNIFTY", "banknifty")
    vix = _context_frame("INDIAVIX", "vix")
    out = panel.merge(n50, on="timestamp", how="left").merge(bank, on="timestamp", how="left").merge(vix, on="timestamp", how="left")

    counts = out.groupby("timestamp")["symbol"].transform("nunique")
    out["breadth_eligible"] = counts >= MIN_STOCKS_PER_TIMESTAMP
    out["breadth_up_1"] = out.groupby("timestamp")["ret_1"].transform(lambda s: (s > 0).mean())
    out["breadth_up_6"] = out.groupby("timestamp")["ret_6"].transform(lambda s: (s > 0).mean())
    out["breadth_up_12"] = out.groupby("timestamp")["ret_12"].transform(lambda s: (s > 0).mean())
    out["breadth_above_fast_ema"] = out.groupby("timestamp")["ema_spread_6_18"].transform(lambda s: (s > 0).mean())

    for col in ["ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "tod_relative_volume", "volatility_12", "overnight_gap"]:
        out[f"xs_pct_{col}"] = out.groupby("timestamp")[col].rank(pct=True, method="average")

    out["rel_to_nifty_6"] = out["ret_6"] - out["nifty_ret_6"]
    out["rel_to_nifty_12"] = out["ret_12"] - out["nifty_ret_12"]
    out["bank_vs_nifty_6"] = out["banknifty_ret_6"] - out["nifty_ret_6"]
    out["vix_change_6"] = out["vix_ret_6"]
    out["market_accel"] = out["nifty_ret_1"] - out["nifty_ret_6"] / 6.0
    return out.replace([np.inf, -np.inf], np.nan)


def add_executable_labels(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("timestamp").copy()
        g["entry_open"] = g["Open"].shift(-1)
        g["entry_ts"] = g["timestamp"].shift(-1)
        entry_day = g["entry_ts"].map(lambda x: x.tz_convert(IST).date() if pd.notna(x) else None)
        for h, bars in HORIZONS.items():
            g[f"exit_open_{h}"] = g["Open"].shift(-(bars + 1))
            g[f"exit_ts_{h}"] = g["timestamp"].shift(-(bars + 1))
            exit_day = g[f"exit_ts_{h}"].map(lambda x: x.tz_convert(IST).date() if pd.notna(x) else None)
            gross = g[f"exit_open_{h}"] / g["entry_open"] - 1.0
            same_day = entry_day == exit_day
            g[f"gross_{h}"] = gross.where(same_day)
            g[f"net_{h}"] = g[f"gross_{h}"].map(lambda z: _round_trip_net(float(z)) if pd.notna(z) else np.nan)
            g[f"profit_{h}"] = (g[f"net_{h}"] > 0).astype("Int64").where(g[f"net_{h}"].notna())
        rows.append(g)
    return pd.concat(rows, ignore_index=True).sort_values(["timestamp", "symbol"])


def add_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    dates = sorted(out["date"].dropna().unique())
    if len(dates) < 15:
        raise RuntimeError("Too few sessions for train/validation/test partitioning")
    train_end_i = max(0, int(len(dates) * 0.60) - 1)
    val_end_i = max(train_end_i + 1, int(len(dates) * 0.80) - 1)
    train_end = dates[train_end_i]
    val_end = dates[min(val_end_i, len(dates) - 2)]
    out["split"] = np.select(
        [out["date"] <= train_end, out["date"] <= val_end],
        ["TRAIN", "VALIDATION"],
        default="TEST",
    )
    meta = {
        "sessions": len(dates),
        "first_session": dates[0],
        "last_session": dates[-1],
        "train_end": train_end,
        "validation_end": val_end,
        "train_sessions": len([d for d in dates if d <= train_end]),
        "validation_sessions": len([d for d in dates if train_end < d <= val_end]),
        "test_sessions": len([d for d in dates if d > val_end]),
    }
    return out, meta


def main() -> None:
    print("Share-Trading-AI v29A3 Canonical Dataset Builder")
    print("NO BROKER ORDERS ARE SENT. Data engineering / research preparation only.")
    print(f"Provider store: {PROVIDER} | interval: {INTERVAL}")
    print("Labels: signal at bar t -> enter next open -> exit future open; Kotak-profile costs + 2 bps/side slippage.")

    panel = build_equity_panel()
    panel = add_market_context(panel)
    panel = add_executable_labels(panel)
    panel, split_meta = add_splits(panel)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = OUT_DIR / f"v29a3_canonical_{PROVIDER}_{INTERVAL}.csv.gz"
    schema_path = OUT_DIR / "v29a3_schema.json"
    quality_path = OUT_DIR / "v29a3_quality_summary.json"

    panel.to_csv(dataset_path, index=False, compression="gzip")

    feature_cols = [c for c in panel.columns if c not in {
        "timestamp", "symbol", "date", "clock", "session", "split", "entry_ts", "entry_open"
    } and not c.startswith("exit_") and not c.startswith("gross_") and not c.startswith("net_") and not c.startswith("profit_")]
    label_cols = [c for c in panel.columns if c.startswith("net_") or c.startswith("profit_") or c.startswith("gross_")]
    schema = {
        "provider": PROVIDER,
        "interval": INTERVAL,
        "feature_columns": feature_cols,
        "label_columns": label_cols,
        "horizons": HORIZONS,
        "execution_definition": "signal at t; entry Open[t+1]; exit Open[t+1+horizon_bars]; same-day only",
        "cost_model": "Kotak-profile order costs plus 2 bps per side slippage",
        "split": split_meta,
        "prototype_only": PROVIDER == "yahoo",
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    context_cols = ["nifty_close", "banknifty_close", "vix_close"]
    quality = {
        "rows": int(len(panel)),
        "stocks": int(panel["symbol"].nunique()),
        "sessions": int(panel["date"].nunique()),
        "first_timestamp": pd.Timestamp(panel["timestamp"].min()).isoformat(),
        "last_timestamp": pd.Timestamp(panel["timestamp"].max()).isoformat(),
        "duplicate_symbol_timestamp": int(panel.duplicated(["symbol", "timestamp"]).sum()),
        "breadth_eligible_pct": float(panel["breadth_eligible"].mean()),
        "context_nonnull_pct": {c: float(panel[c].notna().mean()) for c in context_cols},
        "split_rows": panel["split"].value_counts().to_dict(),
        "provider_research_grade": PROVIDER != "yahoo",
    }
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print("\nV29A3 CANONICAL DATASET SUMMARY")
    print(f"  rows: {len(panel):,}")
    print(f"  stocks: {panel['symbol'].nunique()}")
    print(f"  sessions: {panel['date'].nunique()} ({panel['date'].min()} -> {panel['date'].max()})")
    print(f"  context coverage: NIFTY={quality['context_nonnull_pct']['nifty_close']:.1%} BANKNIFTY={quality['context_nonnull_pct']['banknifty_close']:.1%} VIX={quality['context_nonnull_pct']['vix_close']:.1%}")
    print(f"  breadth eligible rows: {quality['breadth_eligible_pct']:.1%}")
    print(f"  split sessions: TRAIN={split_meta['train_sessions']} VALIDATION={split_meta['validation_sessions']} TEST={split_meta['test_sessions']}")
    print(f"  split cutoffs: train <= {split_meta['train_end']} | validation <= {split_meta['validation_end']} | test after")
    print(f"  duplicate symbol/timestamps: {quality['duplicate_symbol_timestamp']}")
    print(f"  research grade: {'YES' if PROVIDER != 'yahoo' else 'NO - Yahoo prototype store'}")

    print("\nFILES WRITTEN")
    print(f"  {dataset_path}")
    print(f"  {schema_path}")
    print(f"  {quality_path}")

    print("\nNEXT")
    if PROVIDER == "yahoo":
        print("  Canonical pipeline is now testable end-to-end, but do NOT use this 59-session Yahoo dataset to approve a final trading model.")
        print("  We can use it to test v29B code mechanics only; final model validation waits for multi-year research-grade history.")
    else:
        print("  Research-grade canonical history is ready for the v29B opportunity meta-model and nested OOS validation.")


if __name__ == "__main__":
    main()
