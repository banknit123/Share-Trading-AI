from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_ai.config import DEFAULT_CONFIG
from run_forward_paper_observer_v19 import build_live_panel, feature_block
from run_historical_multimodal_v10 import load_events
from run_intraday_portfolio_v21 import HORIZONS, IST, STARTING_CAPITAL, execution_series, order_costs

OUT_DIR = Path("data/v25")
NOTIONAL = STARTING_CAPITAL * 0.30
MIN_TOTAL_OBS = 24
MIN_BLOCK_OBS = 10
MIN_BLOCK_DAYS = 4


def session_bucket(ts: pd.Timestamp) -> str:
    hhmm = pd.Timestamp(ts).tz_convert(IST).strftime("%H:%M")
    if "09:15" <= hhmm < "11:15":
        return "OPEN"
    if "11:15" <= hhmm < "13:15":
        return "MID"
    if "13:15" <= hhmm <= "15:30":
        return "LATE"
    return "OTHER"


def state_labels(row: pd.Series) -> tuple[str, str, str, str]:
    # Market context: use contemporaneous equal-weight proxy features already in the panel.
    mret = float(row.get("nifty_ret_12", 0.0)) if pd.notna(row.get("nifty_ret_12", np.nan)) else 0.0
    if mret > 0.002:
        market = "MARKET_UP"
    elif mret < -0.002:
        market = "MARKET_DOWN"
    else:
        market = "MARKET_FLAT"

    # Stock trend state: intentionally interpretable, based on features already validated for availability.
    e1 = float(row.get("ema_spread_6_18", 0.0)) if pd.notna(row.get("ema_spread_6_18", np.nan)) else 0.0
    e2 = float(row.get("ema_spread_18_36", 0.0)) if pd.notna(row.get("ema_spread_18_36", np.nan)) else 0.0
    r12 = float(row.get("ret_12", 0.0)) if pd.notna(row.get("ret_12", np.nan)) else 0.0
    if e1 > 0 and e2 > 0 and r12 > 0:
        trend = "UPTREND"
    elif e1 < 0 and e2 < 0 and r12 < 0:
        trend = "DOWNTREND"
    else:
        trend = "MIXED"

    vz = float(row.get("volume_z_36", 0.0)) if pd.notna(row.get("volume_z_36", np.nan)) else 0.0
    vr = float(row.get("volume_ratio_12", 1.0)) if pd.notna(row.get("volume_ratio_12", np.nan)) else 1.0
    volume = "HIGH_VOLUME" if (vz >= 1.0 or vr >= 1.30) else "NORMAL_VOLUME"

    bo = float(row.get("breakout_12", 0.0)) if pd.notna(row.get("breakout_12", np.nan)) else 0.0
    bd = float(row.get("breakdown_12", 0.0)) if pd.notna(row.get("breakdown_12", np.nan)) else 0.0
    pos = float(row.get("position_36", 0.5)) if pd.notna(row.get("position_36", np.nan)) else 0.5
    if bo > 0 or pos >= 0.90:
        structure = "BREAKOUT"
    elif bd > 0 or pos <= 0.10:
        structure = "BREAKDOWN"
    else:
        structure = "RANGE"
    return market, trend, volume, structure


def round_trip_net(entry_value: float, gross_return: float) -> tuple[float, float]:
    sell_value = max(0.0, entry_value * (1.0 + gross_return))
    bz, bk = order_costs(entry_value, "BUY")
    sz, sk = order_costs(sell_value, "SELL")
    return gross_return - (bz + sz) / entry_value, gross_return - (bk + sk) / entry_value


def build_observations(seq: pd.DataFrame) -> pd.DataFrame:
    bars = execution_series(seq)
    usable_cols = [
        "timestamp", "symbol", "Open", "Close", "nifty_ret_12", "ema_spread_6_18",
        "ema_spread_18_36", "ret_12", "volume_z_36", "volume_ratio_12",
        "breakout_12", "breakdown_12", "position_36",
    ]
    panel = seq[[c for c in usable_cols if c in seq.columns]].copy()
    panel = panel.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])
    rows: list[dict] = []

    for symbol in DEFAULT_CONFIG.universe:
        s = bars[symbol]
        if s.empty:
            continue
        lookup = {pd.Timestamp(t): i for i, t in enumerate(s["timestamp"])}
        source = panel[panel["symbol"] == symbol].copy()
        for r in source.itertuples(index=False):
            ts = pd.Timestamp(r.timestamp)
            i = lookup.get(ts)
            if i is None:
                continue
            entry_i = i + 1
            if entry_i >= len(s):
                continue
            entry = s.iloc[entry_i]
            entry_ts = pd.Timestamp(entry["timestamp"])
            if entry_ts.tz_convert(IST).date() != ts.tz_convert(IST).date():
                continue

            rr = source[source["timestamp"] == ts].iloc[0]
            market, trend, volume, structure = state_labels(rr)
            session = session_bucket(entry_ts)
            if session == "OTHER":
                continue

            for horizon, (hbars, _) in HORIZONS.items():
                exit_i = entry_i + hbars
                if exit_i >= len(s):
                    continue
                ex = s.iloc[exit_i]
                exit_ts = pd.Timestamp(ex["timestamp"])
                if exit_ts.tz_convert(IST).date() != entry_ts.tz_convert(IST).date():
                    continue
                entry_px = float(entry["Open"])
                exit_px = float(ex["Open"])
                if entry_px <= 0:
                    continue
                gross = exit_px / entry_px - 1.0
                nz, nk = round_trip_net(NOTIONAL, gross)
                rows.append({
                    "date": entry_ts.tz_convert(IST).strftime("%Y-%m-%d"),
                    "signal_ts": ts,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "symbol": symbol,
                    "horizon": horizon,
                    "session": session,
                    "market_regime": market,
                    "trend_state": trend,
                    "volume_state": volume,
                    "structure_state": structure,
                    "gross_return": gross,
                    "net_zero": nz,
                    "net_kotak": nk,
                })
    return pd.DataFrame(rows)


def block_stats(x: pd.DataFrame) -> dict[str, float]:
    by_day = x.groupby("date")["net_kotak"].mean()
    return {
        "n": int(len(x)),
        "days": int(x["date"].nunique()),
        "mean": float(x["net_kotak"].mean()),
        "median": float(x["net_kotak"].median()),
        "win": float((x["net_kotak"] > 0).mean()),
        "median_day": float(by_day.median()) if len(by_day) else np.nan,
        "positive_days": float((by_day > 0).mean()) if len(by_day) else np.nan,
    }


def evaluate_groups(obs: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(obs["date"].unique())
    split = max(1, len(dates) // 2)
    a_dates = set(dates[:split])
    b_dates = set(dates[split:])

    # Several granularities are tested. More complex groups must still clear minimum support.
    group_specs = [
        ("stock_horizon", ["symbol", "horizon"]),
        ("stock_horizon_session", ["symbol", "horizon", "session"]),
        ("stock_horizon_market", ["symbol", "horizon", "market_regime"]),
        ("stock_horizon_trend", ["symbol", "horizon", "trend_state"]),
        ("stock_horizon_volume", ["symbol", "horizon", "volume_state"]),
        ("stock_horizon_structure", ["symbol", "horizon", "structure_state"]),
        ("stock_horizon_market_trend", ["symbol", "horizon", "market_regime", "trend_state"]),
        ("stock_horizon_session_trend", ["symbol", "horizon", "session", "trend_state"]),
        ("stock_horizon_volume_structure", ["symbol", "horizon", "volume_state", "structure_state"]),
    ]

    out: list[dict] = []
    for level, cols in group_specs:
        for key, g in obs.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            if len(g) < MIN_TOTAL_OBS:
                continue
            a = g[g["date"].isin(a_dates)]
            b = g[g["date"].isin(b_dates)]
            if len(a) < MIN_BLOCK_OBS or len(b) < MIN_BLOCK_OBS:
                continue
            if a["date"].nunique() < MIN_BLOCK_DAYS or b["date"].nunique() < MIN_BLOCK_DAYS:
                continue
            sa, sb, st = block_stats(a), block_stats(b), block_stats(g)

            # Strict approval: positive after-cost expectancy in both halves, majority winners,
            # and positive median day in both halves. This mirrors v24's abstention philosophy.
            approved = (
                sa["mean"] > 0
                and sb["mean"] > 0
                and sa["median_day"] > 0
                and sb["median_day"] > 0
                and sa["win"] > 0.50
                and sb["win"] > 0.50
                and sa["positive_days"] >= 0.55
                and sb["positive_days"] >= 0.55
            )
            record = {
                "level": level,
                "approved": bool(approved),
                "total_n": st["n"],
                "total_days": st["days"],
                "total_mean_net": st["mean"],
                "total_win_rate": st["win"],
                "block_a_n": sa["n"],
                "block_a_mean_net": sa["mean"],
                "block_a_win_rate": sa["win"],
                "block_a_positive_days": sa["positive_days"],
                "block_b_n": sb["n"],
                "block_b_mean_net": sb["mean"],
                "block_b_win_rate": sb["win"],
                "block_b_positive_days": sb["positive_days"],
                "conservative_edge": min(sa["mean"], sb["mean"]),
            }
            for c, v in zip(cols, key):
                record[c] = v
            out.append(record)
    result = pd.DataFrame(out)
    if not result.empty:
        result = result.sort_values(["approved", "conservative_edge", "total_n"], ascending=[False, False, False])
    return result


def main() -> None:
    print("Share-Trading-AI v25 Stock-Horizon-Regime Edge Mapper")
    print("NO BROKER ORDERS ARE SENT. Research only.")
    print("Purpose: discover repeatable after-cost edges; NOT to force trades.")
    print("Approval requires positive Kotak-profile expectancy in BOTH chronological halves.")

    live = build_live_panel(load_events())
    seq, features = feature_block(live)
    obs = build_observations(seq)
    if obs.empty:
        raise RuntimeError("No realizable intraday observations were built")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obs_path = OUT_DIR / "v25_observations.csv"
    obs.to_csv(obs_path, index=False)

    print(f"\nObservations built: {len(obs):,}")
    print(f"Dates: {obs['date'].min()} -> {obs['date'].max()} ({obs['date'].nunique()} sessions)")
    print(f"Stocks: {obs['symbol'].nunique()} | horizons: {sorted(obs['horizon'].unique())}")

    edges = evaluate_groups(obs)
    if edges.empty:
        print("\nNo groups had enough observations for two-block stability testing.")
        return

    edge_path = OUT_DIR / "v25_edge_map.csv"
    approved_path = OUT_DIR / "v25_approved_edges.csv"
    edges.to_csv(edge_path, index=False)
    approved = edges[edges["approved"]].copy()
    approved.to_csv(approved_path, index=False)

    print("\nV25 EDGE MAP SUMMARY")
    print(f"  candidate groups tested: {len(edges)}")
    print(f"  APPROVED robust groups: {len(approved)}")
    print(f"  rejected groups: {len(edges) - len(approved)}")
    print(f"  approval rate: {len(approved)/len(edges):.2%}")

    if approved.empty:
        print("\nNO APPROVED EDGE REGIMES.")
        print("The correct conclusion is that current candle/state definitions have not demonstrated repeatable after-cost edge.")
    else:
        print("\nTOP APPROVED EDGES (up to 30)")
        for i, r in enumerate(approved.head(30).itertuples(index=False), start=1):
            descriptors = []
            for c in ["symbol", "horizon", "session", "market_regime", "trend_state", "volume_state", "structure_state"]:
                if hasattr(r, c) and pd.notna(getattr(r, c)):
                    descriptors.append(f"{c}={getattr(r, c)}")
            print(
                f"  {i:2d}. {r.level:31} {' | '.join(descriptors)} | n={r.total_n} "
                f"edge={r.conservative_edge:+.3%} total_mean={r.total_mean_net:+.3%} "
                f"win={r.total_win_rate:.1%} A={r.block_a_mean_net:+.3%} B={r.block_b_mean_net:+.3%}"
            )

    print("\nBASE STOCK x HORIZON VIEW")
    base = edges[edges["level"] == "stock_horizon"].copy()
    if not base.empty:
        for r in base.sort_values("conservative_edge", ascending=False).itertuples(index=False):
            mark = "APPROVED" if r.approved else "reject"
            print(
                f"  {r.symbol:15} {r.horizon:4} {mark:8} n={r.total_n:4d} "
                f"edge={r.conservative_edge:+.3%} mean={r.total_mean_net:+.3%} win={r.total_win_rate:.1%}"
            )

    print("\nFiles written:")
    print(f"  {obs_path}")
    print(f"  {edge_path}")
    print(f"  {approved_path}")
    print("\nInterpretation: v26 may use ONLY approved rows as candidate regimes; rejected rows remain disabled.")


if __name__ == "__main__":
    main()
