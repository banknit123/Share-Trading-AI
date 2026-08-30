from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_ai.data.market_data import download_history
from run_candle_dominant_v12 import _to_utc, add_candle_features
from run_intraday_portfolio_v21 import IST, STARTING_CAPITAL, order_costs

OUT_DIR = Path("data/v26b")
PERIOD = "60d"
INTERVAL = "5m"
HORIZONS = {"5m": 1, "10m": 2, "30m": 6, "60m": 12, "120m": 24}
NOTIONAL = STARTING_CAPITAL * 0.30
APPROVAL_SLIPPAGE_BPS_PER_SIDE = 2.0
MIN_TOTAL_OBS = 160
MIN_LATE_OBS = 60
MIN_SYMBOLS = 8
MIN_FOLD_OBS = 25

# Broad liquid-NSE research universe. It is intentionally independent of DEFAULT_CONFIG.universe.
# Symbols with unavailable/insufficient data are skipped rather than breaking the study.
RESEARCH_UNIVERSE = (
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS",
    "BHARTIARTL.NS", "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
    "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
    "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS", "SUNPHARMA.NS",
    "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
)


def session_bucket(ts: pd.Timestamp) -> str:
    hhmm = pd.Timestamp(ts).tz_convert(IST).strftime("%H:%M")
    if "09:15" <= hhmm < "11:15":
        return "OPEN"
    if "11:15" <= hhmm < "13:15":
        return "MID"
    if "13:15" <= hhmm <= "15:30":
        return "LATE"
    return "OTHER"


def round_trip_net(gross: float, slippage_bps_side: float) -> float:
    sell_value = max(0.0, NOTIONAL * (1.0 + gross))
    _, buy_kotak = order_costs(NOTIONAL, "BUY")
    _, sell_kotak = order_costs(sell_value, "SELL")
    slip = 2.0 * slippage_bps_side / 10000.0
    return gross - (buy_kotak + sell_kotak) / NOTIONAL - slip


def load_universe() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    print(f"Requested broad research universe: {len(RESEARCH_UNIVERSE)} symbols")
    for symbol in RESEARCH_UNIVERSE:
        try:
            raw = _to_utc(download_history(symbol, period=PERIOD, interval=INTERVAL))
            if raw is None or len(raw) < 500:
                print(f"  {symbol:15} SKIP insufficient bars")
                continue
            f = add_candle_features(raw).sort_index()
            if len(f) < 500:
                print(f"  {symbol:15} SKIP insufficient feature rows")
                continue
            data[symbol] = f
            print(f"  {symbol:15} rows={len(f):5d} latest={f.index.max()}")
        except Exception as exc:
            print(f"  {symbol:15} SKIP {type(exc).__name__}: {exc}")
    if len(data) < MIN_SYMBOLS:
        raise RuntimeError(f"Only {len(data)} usable symbols; need at least {MIN_SYMBOLS}")
    return data


def build_market_proxy(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for symbol, f in data.items():
        q = f[["ret_1", "ret_3", "ret_6", "ret_12", "volatility_12", "vwap_distance"]].copy()
        q["symbol"] = symbol
        pieces.append(q)
    stack = pd.concat(pieces)
    proxy = stack.groupby(level=0).median(numeric_only=True).sort_index()
    proxy = proxy.rename(columns={
        "ret_1": "mkt_ret_1", "ret_3": "mkt_ret_3", "ret_6": "mkt_ret_6",
        "ret_12": "mkt_ret_12", "volatility_12": "mkt_vol_12",
        "vwap_distance": "mkt_vwap_distance",
    })
    return proxy


def add_transferable_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    # Cross-sectional percentile ranks: portable across stocks with very different prices/volatility.
    for src, dst in [
        ("ret_12", "rs_rank"),
        ("volume_z_36", "volume_rank"),
        ("vwap_distance", "vwap_rank"),
        ("volatility_12", "vol_rank"),
    ]:
        out[dst] = out.groupby("timestamp")[src].rank(pct=True, method="average")

    out["relative_ret_12"] = out["ret_12"] - out["mkt_ret_12"]
    out["trend_strength"] = out["ema_spread_6_18"] + out["ema_spread_18_36"]

    out["market_state"] = pd.cut(
        out["mkt_ret_12"],
        bins=[-np.inf, -0.004, -0.0015, 0.0015, 0.004, np.inf],
        labels=["MKT_STRONG_DOWN", "MKT_DOWN", "MKT_FLAT", "MKT_UP", "MKT_STRONG_UP"],
    ).astype(str)

    out["relative_strength"] = pd.cut(
        out["rs_rank"], bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=["RS_BOTTOM20", "RS_WEAK", "RS_MID", "RS_STRONG", "RS_TOP20"],
    ).astype(str)

    cond_up = (out["ema_spread_6_18"] > 0) & (out["ema_spread_18_36"] > 0) & (out["ret_12"] > 0)
    cond_dn = (out["ema_spread_6_18"] < 0) & (out["ema_spread_18_36"] < 0) & (out["ret_12"] < 0)
    out["trend_state"] = np.select([cond_up, cond_dn], ["UPTREND", "DOWNTREND"], default="MIXED")

    out["volume_state"] = np.where(
        (out["volume_z_36"] >= 1.0) | (out["volume_rank"] >= 0.80), "HIGH_VOLUME", "NORMAL_VOLUME"
    )
    out["volatility_state"] = np.select(
        [out["vol_rank"] >= 0.80, out["vol_rank"] <= 0.20],
        ["HIGH_VOL", "LOW_VOL"], default="MID_VOL"
    )
    out["vwap_state"] = np.select(
        [out["vwap_rank"] >= 0.80, out["vwap_rank"] <= 0.20],
        ["ABOVE_VWAP_STRONG", "BELOW_VWAP_STRONG"], default="VWAP_MID"
    )
    out["structure_state"] = np.select(
        [(out["breakout_12"] > 0) | (out["position_36"] >= 0.90),
         (out["breakdown_12"] > 0) | (out["position_36"] <= 0.10)],
        ["BREAKOUT", "BREAKDOWN"], default="RANGE"
    )
    out["session"] = out["timestamp"].map(session_bucket)
    return out


def build_observations(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    proxy = build_market_proxy(data)
    frames = []
    for symbol, f in data.items():
        x = f.copy()
        aligned = proxy.reindex(x.index, method="nearest", tolerance=pd.Timedelta(minutes=3))
        x = x.join(aligned)
        x["symbol"] = symbol
        x["timestamp"] = x.index
        frames.append(x)
    panel = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    panel = add_transferable_states(panel)

    rows: list[dict] = []
    for symbol, s0 in panel.groupby("symbol"):
        s = s0.sort_values("timestamp").reset_index(drop=True)
        for i in range(len(s) - 2):
            signal = s.iloc[i]
            entry_i = i + 1
            entry = s.iloc[entry_i]
            signal_ts = pd.Timestamp(signal["timestamp"])
            entry_ts = pd.Timestamp(entry["timestamp"])
            if session_bucket(entry_ts) == "OTHER":
                continue
            if entry_ts.tz_convert(IST).date() != signal_ts.tz_convert(IST).date():
                continue
            entry_px = float(entry["Open"])
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue

            for horizon, hbars in HORIZONS.items():
                exit_i = entry_i + hbars
                if exit_i >= len(s):
                    continue
                ex = s.iloc[exit_i]
                exit_ts = pd.Timestamp(ex["timestamp"])
                if exit_ts.tz_convert(IST).date() != entry_ts.tz_convert(IST).date():
                    continue
                exit_px = float(ex["Open"])
                gross = exit_px / entry_px - 1.0
                rec = {
                    "date": entry_ts.tz_convert(IST).strftime("%Y-%m-%d"),
                    "symbol": symbol, "signal_ts": signal_ts, "entry_ts": entry_ts,
                    "exit_ts": exit_ts, "horizon": horizon, "gross_return": gross,
                    "net_0bps": round_trip_net(gross, 0.0),
                    "net_1bps": round_trip_net(gross, 1.0),
                    "net_2bps": round_trip_net(gross, 2.0),
                    "net_5bps": round_trip_net(gross, 5.0),
                }
                for c in ["market_state", "relative_strength", "trend_state", "volume_state",
                          "volatility_state", "vwap_state", "structure_state", "session"]:
                    rec[c] = signal[c]
                rows.append(rec)
    return pd.DataFrame(rows)


def stats(x: pd.DataFrame, col: str = "net_2bps") -> dict:
    by_day = x.groupby("date")[col].mean()
    by_symbol = x.groupby("symbol")[col].mean()
    return {
        "n": len(x), "days": x["date"].nunique(), "symbols": x["symbol"].nunique(),
        "mean": float(x[col].mean()), "median": float(x[col].median()),
        "win": float((x[col] > 0).mean()),
        "positive_days": float((by_day > 0).mean()) if len(by_day) else np.nan,
        "positive_symbols": float((by_symbol > 0).mean()) if len(by_symbol) else np.nan,
    }


def evaluate_rules(obs: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(obs["date"].unique())
    cut = dates[int(len(dates) * 0.60)]
    early = obs[obs["date"] < cut]
    late = obs[obs["date"] >= cut]

    symbols = sorted(obs["symbol"].unique())
    fold_map = {s: i % 3 for i, s in enumerate(symbols)}
    obs = obs.copy()
    obs["stock_fold"] = obs["symbol"].map(fold_map)
    early = obs[obs["date"] < cut]
    late = obs[obs["date"] >= cut]

    # No symbol is present in any rule definition. These are deliberately transferable states.
    rule_specs = [
        ("horizon_rs_trend", ["horizon", "relative_strength", "trend_state"]),
        ("horizon_market_rs", ["horizon", "market_state", "relative_strength"]),
        ("horizon_rs_volume", ["horizon", "relative_strength", "volume_state"]),
        ("horizon_rs_vwap", ["horizon", "relative_strength", "vwap_state"]),
        ("horizon_market_trend", ["horizon", "market_state", "trend_state"]),
        ("horizon_session_rs", ["horizon", "session", "relative_strength"]),
        ("horizon_trend_volume", ["horizon", "trend_state", "volume_state"]),
        ("horizon_trend_structure", ["horizon", "trend_state", "structure_state"]),
        ("horizon_rs_volatility", ["horizon", "relative_strength", "volatility_state"]),
        ("horizon_market_rs_trend", ["horizon", "market_state", "relative_strength", "trend_state"]),
        ("horizon_session_rs_trend", ["horizon", "session", "relative_strength", "trend_state"]),
        ("horizon_rs_volume_structure", ["horizon", "relative_strength", "volume_state", "structure_state"]),
        ("horizon_market_rs_vwap", ["horizon", "market_state", "relative_strength", "vwap_state"]),
    ]

    out = []
    for level, cols in rule_specs:
        for key, g in obs.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            if len(g) < MIN_TOTAL_OBS or g["symbol"].nunique() < MIN_SYMBOLS:
                continue
            ge = early.loc[early.index.intersection(g.index)]
            gl = late.loc[late.index.intersection(g.index)]
            if len(gl) < MIN_LATE_OBS or gl["symbol"].nunique() < MIN_SYMBOLS:
                continue
            se, sl, st = stats(ge), stats(gl), stats(g)

            fold_stats = []
            for fold in range(3):
                z = gl[gl["stock_fold"] == fold]
                if len(z) >= MIN_FOLD_OBS and z["symbol"].nunique() >= 2:
                    fold_stats.append(stats(z))
            positive_folds = sum(fs["mean"] > 0 and fs["win"] > 0.50 for fs in fold_stats)

            # Approval is intentionally hard: later-time economics, cross-stock breadth,
            # and 2 bps/side slippage must all remain positive. Early period must not be negative.
            approved = (
                se["mean"] > 0
                and sl["mean"] > 0
                and sl["win"] > 0.52
                and sl["positive_days"] >= 0.55
                and sl["positive_symbols"] >= 0.60
                and len(fold_stats) == 3
                and positive_folds >= 2
            )
            rec = {
                "level": level, "approved": bool(approved),
                "total_n": st["n"], "symbols": st["symbols"], "days": st["days"],
                "early_mean_2bps": se["mean"], "late_mean_2bps": sl["mean"],
                "late_win_rate": sl["win"], "late_positive_days": sl["positive_days"],
                "late_positive_symbols": sl["positive_symbols"],
                "positive_stock_folds": positive_folds, "tested_stock_folds": len(fold_stats),
                "conservative_edge_2bps": min(se["mean"], sl["mean"]),
                "total_mean_0bps": float(g["net_0bps"].mean()),
                "total_mean_1bps": float(g["net_1bps"].mean()),
                "total_mean_2bps": float(g["net_2bps"].mean()),
                "total_mean_5bps": float(g["net_5bps"].mean()),
            }
            for c, v in zip(cols, key):
                rec[c] = v
            out.append(rec)
    result = pd.DataFrame(out)
    if not result.empty:
        result = result.sort_values(["approved", "conservative_edge_2bps", "symbols", "total_n"], ascending=[False, False, False, False])
    return result


def main() -> None:
    print("Share-Trading-AI v26B Market-General Cross-Stock Edge Refinement")
    print("NO BROKER ORDERS ARE SENT. Research only.")
    print("Goal: discover transferable market-state rules, NOT stock-specific rules.")
    print(f"Approval includes Kotak-profile charges + {APPROVAL_SLIPPAGE_BPS_PER_SIDE:.0f} bps/side slippage.")
    print("Rules must survive later dates AND different stock folds.")

    data = load_universe()
    print(f"\nUsable broad universe: {len(data)} stocks")
    proxy = build_market_proxy(data)
    print(f"Market proxy bars: {len(proxy)}")

    obs = build_observations(data)
    if obs.empty:
        raise RuntimeError("No executable observations built")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT_DIR / "v26b_observations.csv", index=False)

    print(f"Observations: {len(obs):,}")
    print(f"Sessions: {obs['date'].min()} -> {obs['date'].max()} ({obs['date'].nunique()})")
    print(f"Stocks represented: {obs['symbol'].nunique()}")

    rules = evaluate_rules(obs)
    if rules.empty:
        print("No transferable rules had enough support for testing.")
        return
    approved = rules[rules["approved"]].copy()
    rules.to_csv(OUT_DIR / "v26b_rule_map.csv", index=False)
    approved.to_csv(OUT_DIR / "v26b_approved_transferable_rules.csv", index=False)

    print("\nV26B MARKET-GENERAL SUMMARY")
    print(f"  transferable candidate rules tested: {len(rules)}")
    print(f"  APPROVED cross-stock rules: {len(approved)}")
    print(f"  approval rate: {len(approved)/len(rules):.2%}")
    print(f"  usable research stocks: {obs['symbol'].nunique()}")

    print("\nTOP APPROVED TRANSFERABLE RULES")
    if approved.empty:
        print("  NONE. Current feature/state framework has not yet demonstrated a market-general edge.")
    else:
        for i, r in enumerate(approved.head(30).itertuples(index=False), 1):
            dims = []
            for c in ["horizon", "market_state", "relative_strength", "trend_state", "volume_state",
                      "volatility_state", "vwap_state", "structure_state", "session"]:
                if hasattr(r, c) and pd.notna(getattr(r, c)):
                    dims.append(f"{c}={getattr(r,c)}")
            print(
                f"  {i:2d}. {r.level:32} {' | '.join(dims)} | n={r.total_n} stocks={r.symbols} "
                f"edge2={r.conservative_edge_2bps:+.3%} late_win={r.late_win_rate:.1%} "
                f"pos_stocks={r.late_positive_symbols:.1%} folds={r.positive_stock_folds}/{r.tested_stock_folds} "
                f"mean0/2/5bps={r.total_mean_0bps:+.3%}/{r.total_mean_2bps:+.3%}/{r.total_mean_5bps:+.3%}"
            )

    print("\nFILES WRITTEN")
    print(f"  {OUT_DIR / 'v26b_observations.csv'}")
    print(f"  {OUT_DIR / 'v26b_rule_map.csv'}")
    print(f"  {OUT_DIR / 'v26b_approved_transferable_rules.csv'}")
    print("\nInterpretation:")
    print("  A rule is useful only if it generalizes across stocks; stock identity is never part of a v26B rule.")
    print("  Zero approved rules is an acceptable result and means the current feature set needs better information, not looser gates.")
    print("  Approved rules are research candidates for a later walk-forward portfolio test, not permission for live trading.")


if __name__ == "__main__":
    main()
