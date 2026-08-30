from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_ai.data.market_data import download_history
from run_candle_dominant_v12 import _to_utc, add_candle_features
from run_intraday_portfolio_v21 import IST, order_costs
from run_market_general_edge_v26b import BROAD_UNIVERSE

OUT_DIR = Path("data/v27a")
HORIZONS = {"5m": 1, "10m": 2, "30m": 6, "60m": 12, "120m": 24}
NOTIONAL = 300000.0
SLIPPAGE_BPS_PER_SIDE = 2.0
MIN_STOCKS_PER_TS = 20


def _session_bucket(ts: pd.Timestamp) -> str:
    hhmm = ts.tz_convert(IST).strftime("%H:%M")
    if "09:15" <= hhmm < "11:15":
        return "OPEN"
    if "11:15" <= hhmm < "13:15":
        return "MID"
    if "13:15" <= hhmm <= "15:30":
        return "LATE"
    return "OTHER"


def _safe_download(symbol: str) -> pd.DataFrame | None:
    try:
        raw = _to_utc(download_history(symbol, period="60d", interval="5m"))
        if raw is None or raw.empty:
            return None
        return raw
    except Exception as exc:
        print(f"{symbol:15} SKIP {type(exc).__name__}: {exc}")
        return None


def _round_trip_net(gross: float) -> float:
    sell_value = max(0.0, NOTIONAL * (1.0 + gross))
    _, buy_cost = order_costs(NOTIONAL, "BUY")
    _, sell_cost = order_costs(sell_value, "SELL")
    slip = 2.0 * SLIPPAGE_BPS_PER_SIDE / 10000.0
    return gross - (buy_cost + sell_cost) / NOTIONAL - slip


def build_base_panel() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in BROAD_UNIVERSE:
        raw = _safe_download(symbol)
        if raw is None:
            continue
        f = add_candle_features(raw).sort_index()
        f["symbol"] = symbol
        f["timestamp"] = f.index
        f["date"] = pd.Series(f.index, index=f.index).map(lambda x: x.tz_convert(IST).strftime("%Y-%m-%d"))
        f["session"] = pd.Series(f.index, index=f.index).map(_session_bucket)
        frames.append(f)
        print(f"{symbol:15} rows={len(f):5d} latest={pd.Timestamp(f.index.max()).isoformat()}")

    if not frames:
        raise RuntimeError("No usable broad-universe stocks were downloaded")
    panel = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    panel = panel[panel["session"] != "OTHER"].sort_values(["timestamp", "symbol"])
    print(f"\nUsable broad universe: {panel['symbol'].nunique()} stocks")
    return panel


def add_market_and_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    counts = out.groupby("timestamp")["symbol"].transform("nunique")
    out = out[counts >= MIN_STOCKS_PER_TS].copy()

    def xrank(col: str) -> pd.Series:
        return out.groupby("timestamp")[col].rank(pct=True, method="average")

    for col in ["ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "volume_ratio_12", "volume_z_36", "vwap_distance", "volatility_12", "volatility_36"]:
        if col in out.columns:
            out[f"xs_pct_{col}"] = xrank(col)

    market = out.groupby("timestamp").agg(
        market_ret_1=("ret_1", "median"),
        market_ret_3=("ret_3", "median"),
        market_ret_6=("ret_6", "median"),
        market_ret_12=("ret_12", "median"),
        market_vol_12=("volatility_12", "median"),
        market_vwap_dist=("vwap_distance", "median"),
        breadth_up_1=("ret_1", lambda x: float((x > 0).mean())),
        breadth_up_6=("ret_6", lambda x: float((x > 0).mean())),
        breadth_above_vwap=("vwap_distance", lambda x: float((x > 0).mean())),
        breadth_uptrend=("ema_spread_6_18", lambda x: float((x > 0).mean())),
    ).reset_index()
    market["market_accel_6"] = market["market_ret_1"] - market["market_ret_6"] / 6.0
    out = out.merge(market, on="timestamp", how="left")

    out["rel_ret_1"] = out["ret_1"] - out["market_ret_1"]
    out["rel_ret_3"] = out["ret_3"] - out["market_ret_3"]
    out["rel_ret_6"] = out["ret_6"] - out["market_ret_6"]
    out["rel_ret_12"] = out["ret_12"] - out["market_ret_12"]
    out["trend_quality"] = (
        np.sign(out["ema_spread_6_18"]) + np.sign(out["ema_spread_18_36"]) + np.sign(out["ret_12"])
    ) / 3.0
    out["vol_expansion"] = out["volatility_12"] / out["volatility_36"].replace(0, np.nan)
    out["opening_strength"] = out.groupby(["symbol", "date"])["Close"].transform(lambda x: x / x.iloc[0] - 1.0)
    return out.replace([np.inf, -np.inf], np.nan)


def add_executable_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for symbol, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("timestamp").copy()
        g["entry_open"] = g["Open"].shift(-1)
        g["entry_ts"] = g["timestamp"].shift(-1)
        for h, bars in HORIZONS.items():
            g[f"exit_open_{h}"] = g["Open"].shift(-(bars + 1))
            g[f"exit_ts_{h}"] = g["timestamp"].shift(-(bars + 1))
            gross = g[f"exit_open_{h}"] / g["entry_open"] - 1.0
            same_day = (
                g["entry_ts"].map(lambda x: x.tz_convert(IST).date() if pd.notna(x) else None)
                == g[f"exit_ts_{h}"].map(lambda x: x.tz_convert(IST).date() if pd.notna(x) else None)
            )
            g[f"gross_{h}"] = gross.where(same_day)
            g[f"net_{h}"] = g[f"gross_{h}"].map(lambda x: _round_trip_net(float(x)) if pd.notna(x) else np.nan)
        rows.append(g)
    return pd.concat(rows).sort_values(["timestamp", "symbol"])


def feature_candidates(df: pd.DataFrame) -> list[str]:
    wanted = [
        "rel_ret_1", "rel_ret_3", "rel_ret_6", "rel_ret_12",
        "xs_pct_ret_1", "xs_pct_ret_3", "xs_pct_ret_6", "xs_pct_ret_12", "xs_pct_ret_24",
        "xs_pct_volume_ratio_12", "xs_pct_volume_z_36", "xs_pct_vwap_distance",
        "xs_pct_volatility_12", "xs_pct_volatility_36",
        "breadth_up_1", "breadth_up_6", "breadth_above_vwap", "breadth_uptrend",
        "market_ret_1", "market_ret_6", "market_ret_12", "market_accel_6", "market_vol_12",
        "vwap_distance", "ema_spread_6_18", "ema_spread_18_36", "rsi_14",
        "volume_ratio_12", "volume_z_36", "volatility_12", "volatility_36", "vol_expansion",
        "position_36", "breakout_12", "breakdown_12", "trend_quality", "opening_strength",
    ]
    return [c for c in wanted if c in df.columns]


def _bucket_stats(x: pd.DataFrame, outcome: str) -> dict[str, float]:
    if x.empty:
        return {"n": 0, "stocks": 0, "days": 0, "mean": np.nan, "median": np.nan, "win": np.nan, "positive_days": np.nan}
    by_day = x.groupby("date")[outcome].mean()
    return {
        "n": len(x),
        "stocks": x["symbol"].nunique(),
        "days": x["date"].nunique(),
        "mean": x[outcome].mean(),
        "median": x[outcome].median(),
        "win": (x[outcome] > 0).mean(),
        "positive_days": (by_day > 0).mean() if len(by_day) else np.nan,
    }


def diagnostic_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    dates = sorted(df["date"].dropna().unique())
    d1 = dates[: len(dates) // 3]
    d2 = dates[len(dates) // 3: 2 * len(dates) // 3]
    d3 = dates[2 * len(dates) // 3:]
    blocks = {"A": set(d1), "B": set(d2), "C": set(d3)}

    records: list[dict] = []
    quantiles = [(0.0, 0.2, "Q1"), (0.2, 0.4, "Q2"), (0.4, 0.6, "Q3"), (0.6, 0.8, "Q4"), (0.8, 1.000001, "Q5")]

    for h in HORIZONS:
        outcome = f"net_{h}"
        base = df.dropna(subset=[outcome]).copy()
        for feat in features:
            valid = base.dropna(subset=[feat]).copy()
            if len(valid) < 500:
                continue
            # Global percentile used only to define a simple interpretable feature bucket; stability is assessed later by time block and stock diversity.
            valid["pct"] = valid[feat].rank(pct=True, method="average")
            for lo, hi, label in quantiles:
                q = valid[(valid["pct"] > lo) & (valid["pct"] <= hi)]
                total = _bucket_stats(q, outcome)
                if total["n"] < 100 or total["stocks"] < 10 or total["days"] < 15:
                    continue
                rec = {
                    "horizon": h,
                    "feature": feat,
                    "bucket": label,
                    "total_n": total["n"],
                    "total_stocks": total["stocks"],
                    "total_days": total["days"],
                    "total_mean_net": total["mean"],
                    "total_win": total["win"],
                }
                block_means = []
                block_wins = []
                block_posdays = []
                for name, ds in blocks.items():
                    st = _bucket_stats(q[q["date"].isin(ds)], outcome)
                    rec[f"{name}_n"] = st["n"]
                    rec[f"{name}_stocks"] = st["stocks"]
                    rec[f"{name}_mean_net"] = st["mean"]
                    rec[f"{name}_win"] = st["win"]
                    rec[f"{name}_positive_days"] = st["positive_days"]
                    block_means.append(st["mean"])
                    block_wins.append(st["win"])
                    block_posdays.append(st["positive_days"])
                enough = all(rec[f"{b}_n"] >= 30 and rec[f"{b}_stocks"] >= 8 for b in blocks)
                finite_means = all(pd.notna(x) for x in block_means)
                rec["conservative_edge"] = min(block_means) if finite_means else np.nan
                rec["stable_positive"] = bool(
                    enough
                    and finite_means
                    and all(x > 0 for x in block_means)
                    and all(pd.notna(x) and x > 0.50 for x in block_wins)
                    and all(pd.notna(x) and x >= 0.50 for x in block_posdays)
                )
                records.append(rec)
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["stable_positive", "conservative_edge", "total_n"], ascending=[False, False, False])
    return out


def main() -> None:
    print("Share-Trading-AI v27A Broad-Universe Feature Builder + Edge Diagnostic")
    print("NO BROKER ORDERS ARE SENT. Research only.")
    print("Goal: identify normalized market-wide features that separate profitable from unprofitable executable outcomes.")
    print("Outcomes use next-open -> future-open labels, Kotak-profile costs, and 2 bps/side slippage.")

    base = build_base_panel()
    feat = add_market_and_cross_sectional_features(base)
    full = add_executable_outcomes(feat)
    features = feature_candidates(full)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obs_cols = ["timestamp", "date", "session", "symbol"] + features + [f"net_{h}" for h in HORIZONS]
    full[obs_cols].to_csv(OUT_DIR / "v27a_feature_observations.csv", index=False)

    diag = diagnostic_table(full, features)
    diag.to_csv(OUT_DIR / "v27a_feature_diagnostics.csv", index=False)
    stable = diag[diag["stable_positive"]].copy() if not diag.empty else pd.DataFrame()
    stable.to_csv(OUT_DIR / "v27a_stable_feature_buckets.csv", index=False)

    print(f"\nRows: {len(full):,}")
    print(f"Sessions: {full['date'].min()} -> {full['date'].max()} ({full['date'].nunique()})")
    print(f"Stocks represented: {full['symbol'].nunique()}")
    print(f"Features diagnosed: {len(features)}")
    print("\nV27A FEATURE DIAGNOSTIC SUMMARY")
    print(f"  feature x horizon x quintile buckets tested: {len(diag)}")
    print(f"  stable positive buckets: {len(stable)}")

    print("\nTOP STABLE FEATURE BUCKETS (up to 30)")
    if stable.empty:
        print("  NONE. Richer features still did not show stable positive executable edge individually.")
    else:
        for i, r in enumerate(stable.head(30).itertuples(index=False), start=1):
            print(
                f"  {i:2d}. {r.horizon:4} {r.feature:28} {r.bucket} "
                f"edge={r.conservative_edge:+.3%} mean={r.total_mean_net:+.3%} "
                f"win={r.total_win:.1%} n={r.total_n} stocks={r.total_stocks} days={r.total_days} "
                f"A={r.A_mean_net:+.3%} B={r.B_mean_net:+.3%} C={r.C_mean_net:+.3%}"
            )

    print("\nFILES WRITTEN")
    print(f"  {OUT_DIR / 'v27a_feature_observations.csv'}")
    print(f"  {OUT_DIR / 'v27a_feature_diagnostics.csv'}")
    print(f"  {OUT_DIR / 'v27a_stable_feature_buckets.csv'}")
    print("\nInterpretation:")
    print("  Stable buckets are hypotheses for v27B meta-labeling, not permission to trade.")
    print("  If none survive, the next step is better data/information, not looser gates.")


if __name__ == "__main__":
    main()
