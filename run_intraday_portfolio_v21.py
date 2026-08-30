from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from trading_ai.config import DEFAULT_CONFIG
from run_forward_paper_observer_v19 import build_live_panel, feature_block
from run_historical_multimodal_v10 import load_events

REPLAY_DATE = "2026-08-28"
IST = "Asia/Kolkata"
UTC = "UTC"
STARTING_CAPITAL = 1_000_000.0
MAX_POSITIONS = 3
MIN_COMBINED_SCORE = 0.15
MIN_HOLD_BARS = 2          # 10 minutes before ordinary rotation
REBALANCE_BARS = 1         # every 5 minutes

# Multi-horizon forecast structure: faster windows time entries/exits, slower windows set conviction.
HORIZONS = {
    "5m": (1, 0.10),
    "10m": (2, 0.15),
    "30m": (6, 0.30),
    "60m": (12, 0.25),
    "120m": (24, 0.20),
}

# Current NSE cash-market statutory rates used for the 2026 replay.
# Cash-market transaction outflow: Rs 307/crore each side = 0.00307%.
NSE_TXN_RATE = 307.0 / 10_000_000.0
SEBI_RATE = 10.0 / 10_000_000.0
STAMP_INTRADAY_BUY = 0.00003       # 0.003% buy side
STT_INTRADAY_SELL = 0.00025        # 0.025% sell side
GST_RATE = 0.18


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    entry_ts: pd.Timestamp
    entry_bar: int
    buy_cost_zero: float
    buy_cost_kotak: float


def brokerage_zero(order_value: float) -> float:
    return 0.0


def brokerage_kotak_trade_free(order_value: float) -> float:
    # Published cash-intraday charge after introductory period: Rs 10/order or 0.05%, lower.
    return min(10.0, 0.0005 * order_value)


def side_charges(order_value: float, side: str, brokerage: float) -> float:
    txn = order_value * NSE_TXN_RATE
    sebi = order_value * SEBI_RATE
    stamp = order_value * STAMP_INTRADAY_BUY if side == "BUY" else 0.0
    stt = order_value * STT_INTRADAY_SELL if side == "SELL" else 0.0
    gst = GST_RATE * (brokerage + txn + sebi)
    return brokerage + txn + sebi + stamp + stt + gst


def order_costs(order_value: float, side: str) -> tuple[float, float]:
    zero = side_charges(order_value, side, brokerage_zero(order_value))
    kotak = side_charges(order_value, side, brokerage_kotak_trade_free(order_value))
    return zero, kotak


def day_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(REPLAY_DATE, tz=IST)
    return start.tz_convert(UTC), (start + pd.Timedelta(days=1)).tz_convert(UTC)


def add_horizon_labels(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy().sort_values(["symbol", "timestamp"])
    g = out.groupby("symbol", sort=False)
    for name, (bars, _) in HORIZONS.items():
        fc = g["Close"].shift(-bars)
        fts = g["timestamp"].shift(-bars)
        raw = fc / out["Close"] - 1.0
        med = raw.groupby(out["timestamp"]).transform("median")
        out[f"target_{name}"] = raw - med
        out[f"future_ts_{name}"] = fts
    return out


def fit_models(train_source: pd.DataFrame, features: list[str], day_start: pd.Timestamp):
    models = {}
    for name in HORIZONS:
        target = f"target_{name}"
        future_ts = f"future_ts_{name}"
        t = train_source.dropna(subset=features + [target, future_ts]).copy()
        t = t[t[future_ts] < day_start]
        if len(t) < 1000:
            raise RuntimeError(f"Too few leakage-safe training rows for {name}: {len(t)}")
        model = HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=260,
            max_leaf_nodes=15,
            min_samples_leaf=70,
            l2_regularization=3.0,
            random_state=42,
        )
        model.fit(t[features], t[target])
        models[name] = model
        print(
            f"  {name:4} train_rows={len(t):6d} latest_signal={t['timestamp'].max()} "
            f"latest_outcome={t[future_ts].max()}"
        )
    return models


def complete_friday_times(seq: pd.DataFrame, features: list[str], day_start, day_end):
    usable = seq.dropna(subset=features).copy()
    friday = usable[(usable["timestamp"] >= day_start) & (usable["timestamp"] < day_end)]
    counts = friday.groupby("timestamp")["symbol"].nunique()
    times = list(counts[counts == len(DEFAULT_CONFIG.universe)].index)
    return friday, times


def execution_series(seq: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for symbol in DEFAULT_CONFIG.universe:
        s = seq[seq["symbol"] == symbol][["timestamp", "Open", "Close"]].copy()
        out[symbol] = s.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out


def next_open(series: pd.DataFrame, signal_ts: pd.Timestamp):
    idx = series.index[series["timestamp"] == signal_ts].to_numpy()
    if len(idx) == 0:
        return None
    p = int(idx[0]) + 1
    if p >= len(series):
        return None
    r = series.iloc[p]
    return pd.Timestamp(r["timestamp"]), float(r["Open"]), p


def zscore_cross_section(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = float(np.std(v))
    if sd <= 1e-12:
        return np.zeros_like(v)
    return (v - float(np.mean(v))) / sd


def rank_snapshot(models, snap: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = snap[["timestamp", "symbol", "Close"]].copy().reset_index(drop=True)
    combined = np.zeros(len(out), dtype=float)
    for name, (_, weight) in HORIZONS.items():
        pred = models[name].predict(snap[features])
        z = zscore_cross_section(pred)
        out[f"pred_{name}"] = pred
        out[f"z_{name}"] = z
        combined += weight * z
    out["combined_score"] = combined
    out["rank"] = out["combined_score"].rank(method="first", ascending=False).astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def target_weights(ranked: pd.DataFrame) -> dict[str, float]:
    eligible = ranked[ranked["combined_score"] >= MIN_COMBINED_SCORE].head(MAX_POSITIONS).copy()
    if eligible.empty:
        return {}
    # Positive confidence weights; cap any single stock at 50% and retain cash when conviction is weak.
    raw = np.maximum(eligible["combined_score"].to_numpy(), 0.01)
    raw = raw / raw.sum()
    raw = np.minimum(raw, 0.50)
    raw = raw / raw.sum()
    # Deploy up to 90% capital; always retain 10% cash buffer.
    weights = 0.90 * raw
    return {str(s): float(w) for s, w in zip(eligible["symbol"], weights)}


def portfolio_value(cash: float, positions: dict[str, Position], prices: dict[str, float]) -> float:
    return cash + sum(p.qty * prices.get(sym, p.entry_price) for sym, p in positions.items())


def main() -> None:
    print("Share-Trading-AI v21 continuous multi-horizon intraday portfolio replay")
    print(f"Replay date: {REPLAY_DATE} | starting capital: INR {STARTING_CAPITAL:,.0f}")
    print("NO BROKER ORDERS ARE SENT. Historical replay only.")
    print("Decision cadence: every 5 minutes; forecasts: 5m/10m/30m/60m/120m.")
    print("Portfolio: up to 3 concurrent long positions; next-bar-open execution; capital rotates intraday.")
    print("Cost reports: ZERO-BROKERAGE statutory profile + Kotak Trade Free published brokerage profile.")

    live = build_live_panel(load_events())
    seq, features = feature_block(live)
    labeled = add_horizon_labels(seq)
    day_start, day_end = day_bounds()

    print("\nLeakage-safe model training:")
    models = fit_models(labeled, features, day_start)

    friday, times = complete_friday_times(seq, features, day_start, day_end)
    if len(times) < 10:
        raise RuntimeError(f"Too few complete Friday snapshots: {len(times)}")
    bars = execution_series(seq)
    print(f"\nFriday complete 10-stock bars: {len(times)} | {min(times)} -> {max(times)}")

    cash_zero = STARTING_CAPITAL
    cash_kotak = STARTING_CAPITAL
    positions: dict[str, Position] = {}
    trade_log = []
    equity_curve = []
    checkpoint_rows = []

    # The same holdings are used under both cost profiles; only cost accounting differs.
    for step, ts in enumerate(times[::REBALANCE_BARS]):
        snap = friday[friday["timestamp"] == ts].copy().sort_values("symbol")
        if snap["symbol"].nunique() != len(DEFAULT_CONFIG.universe):
            continue
        ranked = rank_snapshot(models, snap, features)
        desired = target_weights(ranked)

        # Execution prices are the next 5-minute OPEN; skip if unavailable.
        exec_info = {}
        for symbol in DEFAULT_CONFIG.universe:
            n = next_open(bars[symbol], pd.Timestamp(ts))
            if n is not None:
                exec_info[symbol] = n
        if len(exec_info) != len(DEFAULT_CONFIG.universe):
            continue

        exec_ts = min(v[0] for v in exec_info.values())
        prices = {s: v[1] for s, v in exec_info.items()}
        bar_pos = {s: v[2] for s, v in exec_info.items()}

        pre_value_zero = portfolio_value(cash_zero, positions, prices)
        pre_value_kotak = portfolio_value(cash_kotak, positions, prices)

        # SELL positions that have fallen out of desired set after minimum hold.
        for symbol in list(positions):
            p = positions[symbol]
            held_bars = bar_pos[symbol] - p.entry_bar
            should_exit = symbol not in desired and held_bars >= MIN_HOLD_BARS
            if not should_exit:
                continue
            px = prices[symbol]
            value = p.qty * px
            sell_zero, sell_kotak = order_costs(value, "SELL")
            cash_zero += value - sell_zero
            cash_kotak += value - sell_kotak
            gross = px / p.entry_price - 1.0
            cost_zero = p.buy_cost_zero + sell_zero
            cost_kotak = p.buy_cost_kotak + sell_kotak
            trade_log.append({
                "symbol": symbol, "entry_ts": p.entry_ts, "exit_ts": exec_ts,
                "entry": p.entry_price, "exit": px, "qty": p.qty,
                "gross_return": gross,
                "cost_zero": cost_zero, "cost_kotak": cost_kotak,
                "net_pnl_zero": p.qty * (px - p.entry_price) - cost_zero,
                "net_pnl_kotak": p.qty * (px - p.entry_price) - cost_kotak,
            })
            del positions[symbol]

        # Buy missing desired names toward target allocation. Existing positions are not churned merely to resize.
        cur_value_zero = portfolio_value(cash_zero, positions, prices)
        for symbol, w in desired.items():
            if symbol in positions:
                continue
            target_value = cur_value_zero * w
            affordable = max(0.0, min(target_value, cash_zero * 0.98))
            px = prices[symbol]
            qty = int(affordable // px)
            if qty <= 0:
                continue
            order_value = qty * px
            buy_zero, buy_kotak = order_costs(order_value, "BUY")
            if order_value + buy_zero > cash_zero or order_value + buy_kotak > cash_kotak:
                continue
            cash_zero -= order_value + buy_zero
            cash_kotak -= order_value + buy_kotak
            positions[symbol] = Position(
                symbol=symbol, qty=qty, entry_price=px, entry_ts=exec_ts,
                entry_bar=bar_pos[symbol], buy_cost_zero=buy_zero, buy_cost_kotak=buy_kotak,
            )

        post_zero = portfolio_value(cash_zero, positions, prices)
        post_kotak = portfolio_value(cash_kotak, positions, prices)
        equity_curve.append((exec_ts, post_zero, post_kotak, len(positions)))

        checkpoint_rows.append({
            "signal_ts": ts,
            "exec_ts": exec_ts,
            "rank1": str(ranked.iloc[0]["symbol"]),
            "score1": float(ranked.iloc[0]["combined_score"]),
            "desired": ",".join(desired.keys()) if desired else "CASH",
            "positions": ",".join(sorted(positions.keys())) if positions else "CASH",
            "equity_zero": post_zero,
            "equity_kotak": post_kotak,
        })

    # Force square-off at the last available common next-open / latest Friday open.
    if positions:
        last_ts = times[-1]
        for symbol in list(positions):
            s = bars[symbol]
            day_rows = s[(s["timestamp"] >= day_start) & (s["timestamp"] < day_end)]
            if day_rows.empty:
                continue
            r = day_rows.iloc[-1]
            px = float(r["Open"])
            exit_ts = pd.Timestamp(r["timestamp"])
            p = positions[symbol]
            value = p.qty * px
            sell_zero, sell_kotak = order_costs(value, "SELL")
            cash_zero += value - sell_zero
            cash_kotak += value - sell_kotak
            gross = px / p.entry_price - 1.0
            trade_log.append({
                "symbol": symbol, "entry_ts": p.entry_ts, "exit_ts": exit_ts,
                "entry": p.entry_price, "exit": px, "qty": p.qty,
                "gross_return": gross,
                "cost_zero": p.buy_cost_zero + sell_zero,
                "cost_kotak": p.buy_cost_kotak + sell_kotak,
                "net_pnl_zero": p.qty * (px - p.entry_price) - (p.buy_cost_zero + sell_zero),
                "net_pnl_kotak": p.qty * (px - p.entry_price) - (p.buy_cost_kotak + sell_kotak),
            })
            del positions[symbol]

    trades = pd.DataFrame(trade_log)
    checkpoints = pd.DataFrame(checkpoint_rows)

    print("\nV21 DECISION TRACE (every 5m; first 20 shown)")
    for r in checkpoints.head(20).itertuples(index=False):
        print(
            f"  signal={r.signal_ts} exec={r.exec_ts} rank1={r.rank1:13} "
            f"score={r.score1:+.3f} desired=[{r.desired}] held=[{r.positions}]"
        )
    if len(checkpoints) > 20:
        print(f"  ... {len(checkpoints)-20} additional checkpoints omitted from display")

    print("\nV21 TRADE LOG")
    if trades.empty:
        print("  No trades were generated by the frozen decision rules.")
    else:
        for i, r in enumerate(trades.itertuples(index=False), start=1):
            print(
                f"  {i:02d} {r.symbol:13} BUY={r.entry_ts} @{r.entry:.2f} "
                f"SELL={r.exit_ts} @{r.exit:.2f} qty={r.qty} gross={r.gross_return:+.3%} "
                f"PnL_zero=INR {r.net_pnl_zero:+.2f} PnL_kotak=INR {r.net_pnl_kotak:+.2f}"
            )

    final_zero = cash_zero
    final_kotak = cash_kotak
    ret_zero = final_zero / STARTING_CAPITAL - 1.0
    ret_kotak = final_kotak / STARTING_CAPITAL - 1.0

    print("\nV21 FRIDAY PORTFOLIO SUMMARY")
    print(f"  starting capital: INR {STARTING_CAPITAL:,.2f}")
    print(f"  completed trades: {len(trades)}")
    if not trades.empty:
        print(f"  gross winning trades: {(trades['gross_return'] > 0).mean():.2%}")
        print(f"  profitable trades after ZERO-brokerage statutory costs: {(trades['net_pnl_zero'] > 0).mean():.2%}")
        print(f"  profitable trades under Kotak Trade Free profile: {(trades['net_pnl_kotak'] > 0).mean():.2%}")
        print(f"  total statutory costs (zero brokerage): INR {trades['cost_zero'].sum():,.2f}")
        print(f"  total costs incl Kotak brokerage: INR {trades['cost_kotak'].sum():,.2f}")
    print(f"  ZERO-brokerage final capital: INR {final_zero:,.2f} | return={ret_zero:+.3%}")
    print(f"  KOTAK-profile final capital: INR {final_kotak:,.2f} | return={ret_kotak:+.3%}")

    if equity_curve:
        eq = pd.DataFrame(equity_curve, columns=["ts", "zero", "kotak", "positions"])
        peak_zero = eq["zero"].cummax()
        dd_zero = eq["zero"] / peak_zero - 1.0
        peak_kotak = eq["kotak"].cummax()
        dd_kotak = eq["kotak"] / peak_kotak - 1.0
        print(f"  max intraday drawdown (zero brokerage): {dd_zero.min():+.3%}")
        print(f"  max intraday drawdown (Kotak profile): {dd_kotak.min():+.3%}")

    print("\nInterpretation: v21 is a one-day historical portfolio replay. It is not live-trading approval.")
    print("The next research step is to replay the exact frozen v21 rules across many prior sessions without retuning them day-by-day.")


if __name__ == "__main__":
    main()
