from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Compatibility alias for the v27A module's broad-universe import.
import run_market_general_edge_v26b as v26b
if not hasattr(v26b, "BROAD_UNIVERSE"):
    v26b.BROAD_UNIVERSE = v26b.RESEARCH_UNIVERSE

import run_broad_feature_diagnostic_v27a as v27a
from run_intraday_portfolio_v21 import IST

OUT_DIR = Path("data/v28a")
HORIZONS = v27a.HORIZONS
MIN_EVENT_OBS = 90
MIN_BLOCK_OBS = 20
MIN_STOCKS = 8
MIN_DAYS = 10

# Broad-sector map used only for contemporaneous sector context. Missing names fall back to OTHER.
SECTOR_MAP = {
    "HDFCBANK.NS":"BANK", "ICICIBANK.NS":"BANK", "KOTAKBANK.NS":"BANK", "SBIN.NS":"BANK", "INDUSINDBK.NS":"BANK",
    "BAJFINANCE.NS":"FIN", "BAJAJFINSV.NS":"FIN", "HDFCLIFE.NS":"FIN", "SBILIFE.NS":"FIN", "SHRIRAMFIN.NS":"FIN",
    "TCS.NS":"IT", "INFY.NS":"IT", "HCLTECH.NS":"IT", "TECHM.NS":"IT", "WIPRO.NS":"IT",
    "RELIANCE.NS":"ENERGY", "ONGC.NS":"ENERGY", "NTPC.NS":"ENERGY", "POWERGRID.NS":"ENERGY", "COALINDIA.NS":"ENERGY",
    "TATAMOTORS.NS":"AUTO", "MARUTI.NS":"AUTO", "M&M.NS":"AUTO", "EICHERMOT.NS":"AUTO", "BAJAJ-AUTO.NS":"AUTO", "HEROMOTOCO.NS":"AUTO",
    "SUNPHARMA.NS":"PHARMA", "DRREDDY.NS":"PHARMA", "CIPLA.NS":"PHARMA", "APOLLOHOSP.NS":"PHARMA",
    "HINDUNILVR.NS":"CONSUMER", "ITC.NS":"CONSUMER", "NESTLEIND.NS":"CONSUMER", "TATACONSUM.NS":"CONSUMER",
    "ASIANPAINT.NS":"CONSUMER", "TITAN.NS":"CONSUMER", "TRENT.NS":"CONSUMER",
    "TATASTEEL.NS":"METALS", "JSWSTEEL.NS":"METALS", "HINDALCO.NS":"METALS",
    "LT.NS":"INDUSTRIAL", "BEL.NS":"INDUSTRIAL", "ADANIENT.NS":"INDUSTRIAL", "ADANIPORTS.NS":"INDUSTRIAL", "GRASIM.NS":"INDUSTRIAL", "ULTRACEMCO.NS":"INDUSTRIAL",
    "BHARTIARTL.NS":"TELECOM",
}


def add_event_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy().sort_values(["symbol", "timestamp"])
    out["sector"] = out["symbol"].map(SECTOR_MAP).fillna("OTHER")
    out["minute_key"] = out["timestamp"].map(lambda x: pd.Timestamp(x).tz_convert(IST).strftime("%H:%M"))

    # Intraday relative volume versus the stock's own previous observations at the same clock time.
    # shift(1) prevents today's observation from contributing to its own benchmark.
    out["tod_volume_median"] = (
        out.groupby(["symbol", "minute_key"], sort=False)["Volume"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
    )
    out["rvol_tod"] = out["Volume"] / out["tod_volume_median"].replace(0, np.nan)
    out["rvol_tod_pct"] = out.groupby("timestamp")["rvol_tod"].rank(pct=True, method="average")

    # Sector context is contemporaneous and identity-free at decision time.
    sec = out.groupby(["timestamp", "sector"]).agg(
        sector_ret6=("ret_6", "median"),
        sector_ret12=("ret_12", "median"),
        sector_breadth_up=("ret_6", lambda x: float((x > 0).mean())),
        sector_breadth_vwap=("vwap_distance", lambda x: float((x > 0).mean())),
    ).reset_index()
    out = out.merge(sec, on=["timestamp", "sector"], how="left")
    out["sector_relative_ret6"] = out["ret_6"] - out["sector_ret6"]
    out["sector_rs_pct"] = out.groupby("timestamp")["sector_relative_ret6"].rank(pct=True, method="average")

    # Opening-range levels are based only on bars already observed in the same day.
    local_ts = out["timestamp"].map(lambda x: pd.Timestamp(x).tz_convert(IST))
    out["local_time"] = local_ts.map(lambda x: x.strftime("%H:%M"))
    early = out["local_time"] <= "09:45"
    out["or_high_seed"] = out["High"].where(early)
    out["or_low_seed"] = out["Low"].where(early)
    out["or_high"] = out.groupby(["symbol", "date"])["or_high_seed"].cummax()
    out["or_low"] = out.groupby(["symbol", "date"])["or_low_seed"].cummin()
    # Carry the completed opening range forward within the day.
    out["or_high"] = out.groupby(["symbol", "date"])["or_high"].ffill()
    out["or_low"] = out.groupby(["symbol", "date"])["or_low"].ffill()
    prev_close = out.groupby("symbol")["Close"].shift(1)
    out["or_breakout"] = ((prev_close <= out["or_high"]) & (out["Close"] > out["or_high"]) & (out["local_time"] > "09:45"))
    out["or_breakdown"] = ((prev_close >= out["or_low"]) & (out["Close"] < out["or_low"]) & (out["local_time"] > "09:45"))

    # VWAP crossing events rather than simply being above/below VWAP.
    prev_vwap_dist = out.groupby("symbol")["vwap_distance"].shift(1)
    out["vwap_reclaim"] = (prev_vwap_dist <= 0) & (out["vwap_distance"] > 0)
    out["vwap_reject"] = (prev_vwap_dist >= 0) & (out["vwap_distance"] < 0)

    # Cross-sectional percentile features. These are portable across stock price/volatility scales.
    for src, dst in [
        ("ret_6", "rs6_pct"), ("ret_12", "rs12_pct"),
        ("vol_expansion", "vol_expand_pct"), ("trend_quality", "trend_quality_pct"),
        ("opening_strength", "opening_strength_pct"),
    ]:
        if src in out.columns:
            out[dst] = out.groupby("timestamp")[src].rank(pct=True, method="average")

    # Breadth acceleration: current broad participation versus 30 minutes earlier.
    by_ts = out[["timestamp", "breadth_up_6"]].drop_duplicates("timestamp").sort_values("timestamp")
    by_ts["breadth_accel"] = by_ts["breadth_up_6"] - by_ts["breadth_up_6"].shift(6)
    out = out.merge(by_ts[["timestamp", "breadth_accel"]], on="timestamp", how="left")
    return out.replace([np.inf, -np.inf], np.nan)


def event_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    # Quantile-style thresholds are relative, not stock-price-specific constants.
    top_rs = (df["rs6_pct"] >= 0.80) & (df["rs12_pct"] >= 0.70)
    bottom_rs = (df["rs6_pct"] <= 0.20) & (df["rs12_pct"] <= 0.30)
    high_rvol = df["rvol_tod_pct"] >= 0.80
    high_vol_expand = df["vol_expand_pct"] >= 0.80
    strong_trend = df["trend_quality_pct"] >= 0.70
    weak_trend = df["trend_quality_pct"] <= 0.30
    sector_leader = df["sector_rs_pct"] >= 0.75
    sector_laggard = df["sector_rs_pct"] <= 0.25
    market_weak = df["market_ret_6"] < 0
    market_strong = df["market_ret_6"] > 0
    breadth_rising = df["breadth_accel"] > 0
    breadth_falling = df["breadth_accel"] < 0

    return {
        "OR_BREAKOUT_RVOL_RS": df["or_breakout"] & high_rvol & top_rs,
        "OR_BREAKOUT_SECTOR_LEADER": df["or_breakout"] & top_rs & sector_leader,
        "VWAP_RECLAIM_RVOL_RS": df["vwap_reclaim"] & high_rvol & top_rs,
        "VWAP_RECLAIM_TREND": df["vwap_reclaim"] & top_rs & strong_trend,
        "REL_STRENGTH_WEAK_MARKET": top_rs & sector_leader & market_weak,
        "VOL_EXPANSION_MOMENTUM": high_vol_expand & high_rvol & top_rs & strong_trend,
        "BREADTH_CONFIRMED_MOMENTUM": top_rs & sector_leader & breadth_rising & strong_trend,
        "OR_BREAKDOWN_RVOL_WEAK": df["or_breakdown"] & high_rvol & bottom_rs,
        "VWAP_REJECT_WEAK": df["vwap_reject"] & bottom_rs & weak_trend,
        "REL_WEAK_STRONG_MARKET": bottom_rs & sector_laggard & market_strong,
        "VOL_EXPANSION_WEAK": high_vol_expand & high_rvol & bottom_rs & weak_trend,
        "BREADTH_CONFIRMED_WEAK": bottom_rs & sector_laggard & breadth_falling & weak_trend,
    }


def _stats(x: pd.DataFrame, outcome: str) -> dict[str, float]:
    if x.empty:
        return {"n":0,"stocks":0,"days":0,"mean":np.nan,"median":np.nan,"win":np.nan,"positive_days":np.nan,"positive_stocks":np.nan}
    by_day = x.groupby("date")[outcome].mean()
    by_stock = x.groupby("symbol")[outcome].mean()
    return {
        "n": len(x), "stocks": x["symbol"].nunique(), "days": x["date"].nunique(),
        "mean": float(x[outcome].mean()), "median": float(x[outcome].median()),
        "win": float((x[outcome] > 0).mean()),
        "positive_days": float((by_day > 0).mean()),
        "positive_stocks": float((by_stock > 0).mean()),
    }


def evaluate_events(df: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(df["date"].dropna().unique())
    blocks = {
        "A": set(dates[:len(dates)//3]),
        "B": set(dates[len(dates)//3:2*len(dates)//3]),
        "C": set(dates[2*len(dates)//3:]),
    }
    stocks = sorted(df["symbol"].dropna().unique())
    fold_map = {s: i % 3 for i, s in enumerate(stocks)}
    work = df.copy()
    work["stock_fold"] = work["symbol"].map(fold_map)
    masks = event_masks(work)
    records: list[dict] = []

    for event_name, mask in masks.items():
        event_rows = work[mask].copy()
        for horizon in HORIZONS:
            outcome = f"net_{horizon}"
            g = event_rows.dropna(subset=[outcome]).copy()
            if len(g) < MIN_EVENT_OBS or g["symbol"].nunique() < MIN_STOCKS or g["date"].nunique() < MIN_DAYS:
                continue
            rec = {"event": event_name, "horizon": horizon}
            total = _stats(g, outcome)
            rec.update({f"total_{k}":v for k,v in total.items()})
            block_means = []
            block_wins = []
            enough_blocks = True
            for b, ds in blocks.items():
                z = g[g["date"].isin(ds)]
                st = _stats(z, outcome)
                for k,v in st.items(): rec[f"{b}_{k}"] = v
                block_means.append(st["mean"])
                block_wins.append(st["win"])
                if st["n"] < MIN_BLOCK_OBS or st["stocks"] < 4:
                    enough_blocks = False

            fold_means = []
            for fold in range(3):
                z = g[g["stock_fold"] == fold]
                st = _stats(z, outcome)
                rec[f"fold{fold}_n"] = st["n"]
                rec[f"fold{fold}_mean"] = st["mean"]
                rec[f"fold{fold}_win"] = st["win"]
                fold_means.append(st["mean"])

            finite_blocks = all(pd.notna(x) for x in block_means)
            finite_folds = all(pd.notna(x) for x in fold_means)
            rec["conservative_edge"] = min(block_means + fold_means) if finite_blocks and finite_folds else np.nan
            rec["stable_positive"] = bool(
                enough_blocks and finite_blocks and finite_folds
                and all(x > 0 for x in block_means)
                and all(pd.notna(x) and x > 0.50 for x in block_wins)
                and sum(x > 0 for x in fold_means) >= 2
                and total["positive_stocks"] >= 0.55
                and total["positive_days"] >= 0.50
            )
            records.append(rec)

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["stable_positive","conservative_edge","total_n"], ascending=[False,False,False])
    return out


def main() -> None:
    print("Share-Trading-AI v28A Event-Driven Market Opportunity Research Engine")
    print("NO BROKER ORDERS ARE SENT. Research only.")
    print("Candidate generation is event-driven, not every-5-minute-bar driven.")
    print("Outcomes use v27A executable next-open -> future-open labels, costs, and 2 bps/side slippage.")

    base = v27a.build_base_panel()
    broad = v27a.add_market_and_cross_sectional_features(base)
    enriched = add_event_features(broad)
    full = v27a.add_executable_outcomes(enriched)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    masks = event_masks(full)
    candidate_count = pd.DataFrame({name: [int(mask.sum())] for name, mask in masks.items()})
    candidate_count.to_csv(OUT_DIR / "v28a_event_counts.csv", index=False)

    results = evaluate_events(full)
    results.to_csv(OUT_DIR / "v28a_event_results.csv", index=False)
    stable = results[results["stable_positive"]].copy() if not results.empty else pd.DataFrame()
    stable.to_csv(OUT_DIR / "v28a_stable_events.csv", index=False)

    print(f"\nRows evaluated: {len(full):,}")
    print(f"Sessions: {full['date'].min()} -> {full['date'].max()} ({full['date'].nunique()})")
    print(f"Stocks represented: {full['symbol'].nunique()}")
    print(f"Event families: {len(masks)}")
    print(f"Raw event candidates: {sum(int(m.sum()) for m in masks.values()):,}")

    print("\nV28A EVENT-DRIVEN SUMMARY")
    print(f"  event x horizon tests with sufficient support: {len(results)}")
    print(f"  stable positive event/horizon setups: {len(stable)}")

    print("\nTOP STABLE EVENT SETUPS (up to 30)")
    if stable.empty:
        print("  NONE. Current OHLCV-derived event definitions still do not demonstrate stable after-cost market-general edge.")
    else:
        for i, r in enumerate(stable.head(30).itertuples(index=False), start=1):
            print(
                f"  {i:2d}. {r.event:32} {r.horizon:4} "
                f"edge={r.conservative_edge:+.3%} mean={r.total_mean:+.3%} win={r.total_win:.1%} "
                f"n={r.total_n} stocks={r.total_stocks} days={r.total_days} "
                f"A={r.A_mean:+.3%} B={r.B_mean:+.3%} C={r.C_mean:+.3%}"
            )

    print("\nFILES WRITTEN")
    print(f"  {OUT_DIR / 'v28a_event_counts.csv'}")
    print(f"  {OUT_DIR / 'v28a_event_results.csv'}")
    print(f"  {OUT_DIR / 'v28a_stable_events.csv'}")
    print("\nInterpretation:")
    print("  Positive setups are hypotheses for later meta-label/walk-forward validation, not permission to trade.")
    print("  Zero stable setups means richer external/microstructure data should be added rather than loosening gates.")


if __name__ == "__main__":
    main()
