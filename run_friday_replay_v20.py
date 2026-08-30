from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ai.config import DEFAULT_CONFIG
from run_forward_paper_observer_v19 import build_live_panel, feature_block
from run_historical_multimodal_v10 import load_events
from run_walkforward_candle_rank_v18 import EXCESS, TARGET, fit_regressor

FRIDAY_DATE = "2026-08-28"
IST = "Asia/Kolkata"
UTC = "UTC"
HORIZON_BARS = 24          # 120 minutes on 5-minute bars
CHECKPOINT_BARS = 6        # evaluate rankings every 30 minutes
ROUND_TRIP_COST = 0.0012   # same diagnostic assumption used throughout project


def add_strict_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Add 120m raw/excess labels plus the timestamp at which the outcome becomes known."""
    out = panel.copy().sort_values(["symbol", "timestamp"])
    g = out.groupby("symbol", sort=False)
    out["future_close_120m"] = g["Close"].shift(-HORIZON_BARS)
    out["future_ts_120m"] = g["timestamp"].shift(-HORIZON_BARS)
    out[TARGET] = out["future_close_120m"] / out["Close"] - 1.0
    med = out.groupby("timestamp")[TARGET].transform("median")
    out[EXCESS] = out[TARGET] - med
    return out


def replay_day_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ist = pd.Timestamp(FRIDAY_DATE, tz=IST)
    end_ist = start_ist + pd.Timedelta(days=1)
    return start_ist.tz_convert(UTC), end_ist.tz_convert(UTC)


def complete_snapshots(panel: pd.DataFrame, features: list[str], day_start: pd.Timestamp, day_end: pd.Timestamp):
    usable = panel.dropna(subset=features).copy()
    day = usable[(usable["timestamp"] >= day_start) & (usable["timestamp"] < day_end)].copy()
    counts = day.groupby("timestamp")["symbol"].nunique()
    times = list(counts[counts == len(DEFAULT_CONFIG.universe)].index)
    return day, times


def next_bar_execution_map(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for symbol in DEFAULT_CONFIG.universe:
        s = panel[panel["symbol"] == symbol][["timestamp", "Open", "Close"]].copy()
        s = s.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        result[symbol] = s
    return result


def actual_trade_return(series: pd.DataFrame, signal_ts: pd.Timestamp):
    """Enter next bar open and exit 24 bars later at open; return None if Friday lacks enough bars."""
    pos = series.index[series["timestamp"] == signal_ts].to_numpy()
    if len(pos) == 0:
        return None
    signal_pos = int(pos[0])
    entry_pos = signal_pos + 1
    exit_pos = entry_pos + HORIZON_BARS
    if entry_pos >= len(series) or exit_pos >= len(series):
        return None
    entry = series.iloc[entry_pos]
    exit_ = series.iloc[exit_pos]
    return {
        "entry_ts": pd.Timestamp(entry["timestamp"]),
        "exit_ts": pd.Timestamp(exit_["timestamp"]),
        "entry_price": float(entry["Open"]),
        "exit_price": float(exit_["Open"]),
        "gross_return": float(exit_["Open"] / entry["Open"] - 1.0),
    }


def score_snapshot(model, snap: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = snap[["timestamp", "symbol", "Close"]].copy()
    out["score"] = model.predict(snap[features])
    out["model_rank"] = out["score"].rank(method="first", ascending=False).astype(int)
    return out.sort_values("model_rank").reset_index(drop=True)


def main() -> None:
    print("Share-Trading-AI v20 leakage-safe Friday replay")
    print(f"Replay date: {FRIDAY_DATE} (NSE / Asia-Kolkata)")
    print("NO BROKER ORDERS ARE SENT. This is a historical simulation only.")
    print("Frozen predictor: v18/v19 120-minute candle-sequence cross-sectional ranker.")
    print("Execution: signal on completed bar -> BUY rank-1 at next 5m OPEN -> SELL 120m later at OPEN.")
    print(f"Trading-cost assumption: {ROUND_TRIP_COST:.3%} round trip; main P&L uses non-overlapping trades.")

    live = build_live_panel(load_events())
    seq, features = feature_block(live)
    labeled = add_strict_labels(seq)
    day_start, day_end = replay_day_bounds()

    # Strict anti-leakage training: the outcome itself must have become known before Friday began.
    train = labeled.dropna(subset=features + [TARGET, EXCESS, "future_ts_120m"]).copy()
    train = train[train["future_ts_120m"] < day_start].copy()
    if train.empty:
        raise RuntimeError("No pre-Friday labeled training history after leakage guard")

    print(
        f"Training rows={len(train)} | latest training signal={train['timestamp'].max()} "
        f"latest training outcome_ts={train['future_ts_120m'].max()}"
    )
    model = fit_regressor(train, features)

    day, full_times = complete_snapshots(seq, features, day_start, day_end)
    if not full_times:
        raise RuntimeError("No complete 10-stock Friday snapshots were available")

    print(
        f"Friday complete-universe bars={len(full_times)} from {min(full_times)} to {max(full_times)}"
    )

    bars = next_bar_execution_map(seq)

    # Use every 30-minute complete checkpoint for accuracy diagnostics.
    checkpoints = full_times[::CHECKPOINT_BARS]
    rows = []
    for ts in checkpoints:
        snap = day[day["timestamp"] == ts].copy().sort_values("symbol")
        ranked = score_snapshot(model, snap, features)

        actuals = []
        for r in ranked.itertuples(index=False):
            tr = actual_trade_return(bars[str(r.symbol)], pd.Timestamp(ts))
            if tr is None:
                actuals = []
                break
            actuals.append((str(r.symbol), tr))
        if len(actuals) != len(DEFAULT_CONFIG.universe):
            continue

        actual_map = {sym: tr for sym, tr in actuals}
        ranked["actual_gross"] = ranked["symbol"].map(lambda s: actual_map[str(s)]["gross_return"])
        ranked["actual_rank"] = ranked["actual_gross"].rank(method="first", ascending=False).astype(int)
        rank_ic = ranked["score"].corr(ranked["actual_gross"], method="spearman")
        top = ranked.iloc[0]
        med = float(ranked["actual_gross"].median())
        best = ranked.loc[ranked["actual_gross"].idxmax()]
        top_tr = actual_map[str(top.symbol)]

        rows.append({
            "signal_ts": pd.Timestamp(ts),
            "predicted_symbol": str(top.symbol),
            "predicted_actual_rank": int(top.actual_rank),
            "entry_ts": top_tr["entry_ts"],
            "exit_ts": top_tr["exit_ts"],
            "gross": float(top.actual_gross),
            "net": float(top.actual_gross - ROUND_TRIP_COST),
            "median_return": med,
            "excess_vs_median": float(top.actual_gross - med),
            "best_symbol": str(best.symbol),
            "best_return": float(best.actual_gross),
            "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
        })

    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError("No Friday checkpoints had a full 120-minute realizable horizon")

    print("\n30-minute checkpoint accuracy audit (independent overlapping diagnostics):")
    for r in audit.itertuples(index=False):
        print(
            f"  {r.signal_ts} pick={r.predicted_symbol:13} actual_rank={r.predicted_actual_rank:2d}/10 "
            f"gross={r.gross:+.3%} net={r.net:+.3%} excess={r.excess_vs_median:+.3%} "
            f"rank_ic={r.rank_ic:+.3f} best={r.best_symbol}({r.best_return:+.3%})"
        )

    # Main execution simulation: take the first eligible checkpoint, then no new trade until exit.
    executed = []
    next_allowed = None
    for r in audit.itertuples(index=False):
        if next_allowed is not None and pd.Timestamp(r.entry_ts) < next_allowed:
            continue
        executed.append(r)
        next_allowed = pd.Timestamp(r.exit_ts)

    capital = 1.0
    print("\nSIMULATED NON-OVERLAPPING TRADES:")
    for i, r in enumerate(executed, start=1):
        capital *= (1.0 + r.net)
        print(
            f"  trade={i} BUY {r.predicted_symbol:13} entry={r.entry_ts} "
            f"SELL={r.exit_ts} gross={r.gross:+.3%} net={r.net:+.3%} "
            f"actual_rank={r.predicted_actual_rank}/10"
        )

    net_day = capital - 1.0
    profitable = audit["net"] > 0
    top3 = audit["predicted_actual_rank"] <= 3
    exact_top = audit["predicted_actual_rank"] == 1
    positive_excess = audit["excess_vs_median"] > 0

    print("\nFRIDAY ACCURACY SUMMARY")
    print(f"  checkpoints evaluated: {len(audit)}")
    print(f"  exact-best-stock accuracy: {exact_top.mean():.2%}")
    print(f"  top-3 selection accuracy: {top3.mean():.2%}")
    print(f"  profitable-after-cost accuracy: {profitable.mean():.2%}")
    print(f"  outperformed-universe-median accuracy: {positive_excess.mean():.2%}")
    print(f"  mean cross-sectional rank IC: {audit['rank_ic'].mean():+.3f}")
    print(f"  selected-stock mean gross return: {audit['gross'].mean():+.3%}")
    print(f"  selected-stock mean net return: {audit['net'].mean():+.3%}")
    print(f"  selected-stock mean excess vs median: {audit['excess_vs_median'].mean():+.3%}")

    print("\nFRIDAY SIMULATED P&L SUMMARY")
    print(f"  non-overlapping trades: {len(executed)}")
    print(f"  compounded net return: {net_day:+.3%}")
    if executed:
        print(f"  winning trades after costs: {np.mean([r.net > 0 for r in executed]):.2%}")
        print(f"  average executed-trade net return: {np.mean([r.net for r in executed]):+.3%}")

    # Context: what actually happened to the 10-stock universe from the first executable entry
    # to the last executable exit, using the same open-to-open convention.
    first_entry = min(pd.Timestamp(r.entry_ts) for r in executed)
    last_exit = max(pd.Timestamp(r.exit_ts) for r in executed)
    universe_period = []
    for symbol, s in bars.items():
        a = s[s["timestamp"] == first_entry]
        b = s[s["timestamp"] == last_exit]
        if len(a) and len(b):
            rr = float(b.iloc[0]["Open"] / a.iloc[0]["Open"] - 1.0)
            universe_period.append((symbol, rr))
    if universe_period:
        universe_period.sort(key=lambda x: x[1], reverse=True)
        print("\nACTUAL UNIVERSE PERFORMANCE OVER SIMULATED CAPITAL WINDOW")
        for symbol, rr in universe_period:
            print(f"  {symbol:15} {rr:+.3%}")
        print(f"  universe median: {np.median([r for _, r in universe_period]):+.3%}")
        print(f"  best actual stock: {universe_period[0][0]} {universe_period[0][1]:+.3%}")

    print("\nInterpretation note: this is a one-day historical replay, not evidence of future profitability.")


if __name__ == "__main__":
    main()
