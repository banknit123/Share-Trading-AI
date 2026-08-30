from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ai.config import DEFAULT_CONFIG
from run_forward_paper_observer_v19 import build_live_panel, feature_block
from run_historical_multimodal_v10 import load_events
from run_intraday_portfolio_v21 import (
    STARTING_CAPITAL,
    MAX_POSITIONS,
    MIN_COMBINED_SCORE,
    MIN_HOLD_BARS,
    REBALANCE_BARS,
    HORIZONS,
    Position,
    add_horizon_labels,
    fit_models,
    execution_series,
    next_open,
    rank_snapshot,
    target_weights,
    portfolio_value,
    order_costs,
)

IST = "Asia/Kolkata"
UTC = "UTC"
N_SESSIONS = 20


def day_bounds(date_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(date_str, tz=IST)
    return start.tz_convert(UTC), (start + pd.Timedelta(days=1)).tz_convert(UTC)


def complete_day_times(seq: pd.DataFrame, features: list[str], day_start: pd.Timestamp, day_end: pd.Timestamp):
    usable = seq.dropna(subset=features).copy()
    day = usable[(usable["timestamp"] >= day_start) & (usable["timestamp"] < day_end)]
    counts = day.groupby("timestamp")["symbol"].nunique()
    times = list(counts[counts == len(DEFAULT_CONFIG.universe)].index)
    return day, times


def available_complete_sessions(seq: pd.DataFrame, features: list[str]) -> list[str]:
    usable = seq.dropna(subset=features).copy()
    local_dates = usable["timestamp"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d")
    usable = usable.assign(local_date=local_dates)
    counts = usable.groupby(["local_date", "timestamp"])["symbol"].nunique().reset_index(name="n")
    full = counts[counts["n"] == len(DEFAULT_CONFIG.universe)]
    by_day = full.groupby("local_date").size()
    # Require a reasonably complete intraday session; 50 bars = 250 minutes.
    return list(by_day[by_day >= 50].sort_index().index)


def simulate_day(
    date_str: str,
    seq: pd.DataFrame,
    labeled: pd.DataFrame,
    features: list[str],
    bars: dict[str, pd.DataFrame],
    capital_zero: float,
    capital_kotak: float,
) -> dict | None:
    day_start, day_end = day_bounds(date_str)
    day, times = complete_day_times(seq, features, day_start, day_end)
    if len(times) < 10:
        return None

    models = fit_models(labeled, features, day_start)

    cash_zero = float(capital_zero)
    cash_kotak = float(capital_kotak)
    start_zero = cash_zero
    start_kotak = cash_kotak
    positions: dict[str, Position] = {}
    trade_log: list[dict] = []
    equity_zero = [start_zero]
    equity_kotak = [start_kotak]

    for ts in times[::REBALANCE_BARS]:
        snap = day[day["timestamp"] == ts].copy().sort_values("symbol")
        if snap["symbol"].nunique() != len(DEFAULT_CONFIG.universe):
            continue

        ranked = rank_snapshot(models, snap, features)
        desired = target_weights(ranked)

        exec_info = {}
        for symbol in DEFAULT_CONFIG.universe:
            n = next_open(bars[symbol], pd.Timestamp(ts))
            if n is not None and day_start <= n[0] < day_end:
                exec_info[symbol] = n
        if len(exec_info) != len(DEFAULT_CONFIG.universe):
            continue

        exec_ts = min(v[0] for v in exec_info.values())
        prices = {s: v[1] for s, v in exec_info.items()}
        bar_pos = {s: v[2] for s, v in exec_info.items()}

        # Same v21 exit rule: rotate out only after minimum hold when name leaves desired set.
        for symbol in list(positions):
            p = positions[symbol]
            held_bars = bar_pos[symbol] - p.entry_bar
            if symbol in desired or held_bars < MIN_HOLD_BARS:
                continue
            px = prices[symbol]
            value = p.qty * px
            sell_zero, sell_kotak = order_costs(value, "SELL")
            cash_zero += value - sell_zero
            cash_kotak += value - sell_kotak
            trade_log.append({
                "symbol": symbol,
                "entry_ts": p.entry_ts,
                "exit_ts": exec_ts,
                "entry": p.entry_price,
                "exit": px,
                "qty": p.qty,
                "gross_return": px / p.entry_price - 1.0,
                "cost_zero": p.buy_cost_zero + sell_zero,
                "cost_kotak": p.buy_cost_kotak + sell_kotak,
                "net_pnl_zero": p.qty * (px - p.entry_price) - (p.buy_cost_zero + sell_zero),
                "net_pnl_kotak": p.qty * (px - p.entry_price) - (p.buy_cost_kotak + sell_kotak),
            })
            del positions[symbol]

        # Same v21 buy rule: up to 3 names, 90% max deployment, no resize churn.
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
                symbol=symbol,
                qty=qty,
                entry_price=px,
                entry_ts=exec_ts,
                entry_bar=bar_pos[symbol],
                buy_cost_zero=buy_zero,
                buy_cost_kotak=buy_kotak,
            )

        equity_zero.append(portfolio_value(cash_zero, positions, prices))
        equity_kotak.append(portfolio_value(cash_kotak, positions, prices))

    # Force intraday square-off at final Friday/day open, exactly as v21.
    if positions:
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
            trade_log.append({
                "symbol": symbol,
                "entry_ts": p.entry_ts,
                "exit_ts": exit_ts,
                "entry": p.entry_price,
                "exit": px,
                "qty": p.qty,
                "gross_return": px / p.entry_price - 1.0,
                "cost_zero": p.buy_cost_zero + sell_zero,
                "cost_kotak": p.buy_cost_kotak + sell_kotak,
                "net_pnl_zero": p.qty * (px - p.entry_price) - (p.buy_cost_zero + sell_zero),
                "net_pnl_kotak": p.qty * (px - p.entry_price) - (p.buy_cost_kotak + sell_kotak),
            })
            del positions[symbol]

    trades = pd.DataFrame(trade_log)
    end_zero = float(cash_zero)
    end_kotak = float(cash_kotak)

    if trades.empty:
        gross_pnl = 0.0
        cost_zero = 0.0
        cost_kotak = 0.0
        wins_zero = np.nan
        wins_kotak = np.nan
    else:
        cost_zero = float(trades["cost_zero"].sum())
        cost_kotak = float(trades["cost_kotak"].sum())
        net_zero = float(trades["net_pnl_zero"].sum())
        gross_pnl = net_zero + cost_zero
        wins_zero = float((trades["net_pnl_zero"] > 0).mean())
        wins_kotak = float((trades["net_pnl_kotak"] > 0).mean())

    peak_zero = np.maximum.accumulate(np.asarray(equity_zero, dtype=float))
    dd_zero = np.asarray(equity_zero, dtype=float) / peak_zero - 1.0
    peak_kotak = np.maximum.accumulate(np.asarray(equity_kotak, dtype=float))
    dd_kotak = np.asarray(equity_kotak, dtype=float) / peak_kotak - 1.0

    return {
        "date": date_str,
        "start_zero": start_zero,
        "end_zero": end_zero,
        "return_zero": end_zero / start_zero - 1.0,
        "start_kotak": start_kotak,
        "end_kotak": end_kotak,
        "return_kotak": end_kotak / start_kotak - 1.0,
        "trades": int(len(trades)),
        "gross_pnl": gross_pnl,
        "gross_return_on_start": gross_pnl / start_zero if start_zero else 0.0,
        "cost_zero": cost_zero,
        "cost_kotak": cost_kotak,
        "win_zero": wins_zero,
        "win_kotak": wins_kotak,
        "max_dd_zero": float(dd_zero.min()) if len(dd_zero) else 0.0,
        "max_dd_kotak": float(dd_kotak.min()) if len(dd_kotak) else 0.0,
    }


def main() -> None:
    print("Share-Trading-AI v22 frozen multi-day walk-forward portfolio replay")
    print("NO BROKER ORDERS ARE SENT. Historical research only.")
    print("Rules are frozen from v21: same 5/10/30/60/120m weights, threshold, max positions, rotation and costs.")
    print(f"Requested sessions: {N_SESSIONS}")

    live = build_live_panel(load_events())
    seq, features = feature_block(live)
    labeled = add_horizon_labels(seq)
    bars = execution_series(seq)

    sessions = available_complete_sessions(seq, features)
    if len(sessions) < N_SESSIONS:
        print(f"Only {len(sessions)} sufficiently complete sessions are available; using all of them.")
        chosen = sessions
    else:
        chosen = sessions[-N_SESSIONS:]

    if not chosen:
        raise RuntimeError("No complete sessions available for v22 replay")

    print(f"Replay window: {chosen[0]} -> {chosen[-1]} ({len(chosen)} sessions)")

    capital_zero = STARTING_CAPITAL
    capital_kotak = STARTING_CAPITAL
    daily = []

    for i, date_str in enumerate(chosen, start=1):
        print(f"\n===== SESSION {i}/{len(chosen)}: {date_str} =====")
        result = simulate_day(date_str, seq, labeled, features, bars, capital_zero, capital_kotak)
        if result is None:
            print("Skipped: insufficient complete bars")
            continue
        daily.append(result)
        capital_zero = result["end_zero"]
        capital_kotak = result["end_kotak"]
        print(
            f"trades={result['trades']:3d} gross={result['gross_return_on_start']:+.3%} "
            f"zero_net={result['return_zero']:+.3%} kotak_net={result['return_kotak']:+.3%} "
            f"cost_zero=INR {result['cost_zero']:,.2f} cost_kotak=INR {result['cost_kotak']:,.2f}"
        )

    d = pd.DataFrame(daily)
    if d.empty:
        raise RuntimeError("No sessions were successfully replayed")

    total_trades = int(d["trades"].sum())
    total_cost_zero = float(d["cost_zero"].sum())
    total_cost_kotak = float(d["cost_kotak"].sum())
    cumulative_zero = capital_zero / STARTING_CAPITAL - 1.0
    cumulative_kotak = capital_kotak / STARTING_CAPITAL - 1.0

    # Daily equity drawdown using closing capitals.
    eq_zero = np.r_[STARTING_CAPITAL, d["end_zero"].to_numpy(dtype=float)]
    eq_kotak = np.r_[STARTING_CAPITAL, d["end_kotak"].to_numpy(dtype=float)]
    dd_zero = eq_zero / np.maximum.accumulate(eq_zero) - 1.0
    dd_kotak = eq_kotak / np.maximum.accumulate(eq_kotak) - 1.0

    print("\nV22 MULTI-DAY SUMMARY")
    print(f"  sessions replayed: {len(d)}")
    print(f"  total completed trades: {total_trades}")
    print(f"  avg trades/session: {total_trades / len(d):.1f}")
    print(f"  gross-positive sessions: {(d['gross_pnl'] > 0).mean():.2%}")
    print(f"  ZERO-brokerage profitable sessions: {(d['return_zero'] > 0).mean():.2%}")
    print(f"  Kotak-profile profitable sessions: {(d['return_kotak'] > 0).mean():.2%}")
    print(f"  avg gross return/session: {d['gross_return_on_start'].mean():+.3%}")
    print(f"  avg ZERO-brokerage net/session: {d['return_zero'].mean():+.3%}")
    print(f"  avg Kotak-profile net/session: {d['return_kotak'].mean():+.3%}")
    print(f"  total statutory costs: INR {total_cost_zero:,.2f}")
    print(f"  total costs incl Kotak brokerage: INR {total_cost_kotak:,.2f}")
    print(f"  ZERO-brokerage final capital: INR {capital_zero:,.2f} | cumulative_return={cumulative_zero:+.3%}")
    print(f"  KOTAK-profile final capital: INR {capital_kotak:,.2f} | cumulative_return={cumulative_kotak:+.3%}")
    print(f"  max close-to-close drawdown ZERO: {dd_zero.min():+.3%}")
    print(f"  max close-to-close drawdown KOTAK: {dd_kotak.min():+.3%}")
    print(f"  worst ZERO-brokerage session: {d.loc[d['return_zero'].idxmin(), 'date']} {d['return_zero'].min():+.3%}")
    print(f"  best ZERO-brokerage session: {d.loc[d['return_zero'].idxmax(), 'date']} {d['return_zero'].max():+.3%}")

    print("\nDAILY RESULTS")
    for r in d.itertuples(index=False):
        print(
            f"  {r.date} trades={r.trades:3d} gross={r.gross_return_on_start:+.3%} "
            f"zero={r.return_zero:+.3%} kotak={r.return_kotak:+.3%}"
        )

    print("\nInterpretation rule:")
    print("  If gross returns are repeatedly positive but net returns are not, turnover/cost control is the next research target.")
    print("  If gross returns are also unstable/negative, the prediction/portfolio logic itself still lacks robust edge.")
    print("  Do not tune rules day-by-day after seeing these results; use aggregate evidence only.")


if __name__ == "__main__":
    main()
