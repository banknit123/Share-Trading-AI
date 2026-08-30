from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from trading_ai.config import DEFAULT_CONFIG
from run_forward_paper_observer_v19 import build_live_panel, feature_block
from run_historical_multimodal_v10 import load_events
from run_intraday_portfolio_v21 import (
    HORIZONS,
    IST,
    UTC,
    STARTING_CAPITAL,
    MAX_POSITIONS,
    add_horizon_labels,
    day_bounds,
    complete_friday_times,
    execution_series,
    next_open,
    order_costs,
    zscore_cross_section,
)

REPLAY_DATE = "2026-08-28"
MIN_PREDICTED_NET_RETURN = 0.0100   # execute only if predicted net gain >= 1.00%
MIN_RELATIVE_SCORE = 0.50           # strong cross-sectional opportunity only
MIN_HORIZON_AGREEMENT = 3           # at least 3/5 horizons must predict positive raw return
CAPITAL_PER_NEW_TRADE = 0.30        # up to 30% of current equity per new position
CASH_BUFFER = 0.10
MAX_DAILY_ENTRIES = 20

# NSE cash-market windows. These are maximum entry counts, never targets.
SESSION_WINDOWS = [
    ("OPEN", "09:15", "11:15", 10),
    ("MID",  "11:15", "13:15", 5),
    ("LATE", "13:15", "15:15", 5),
]


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    entry_ts: pd.Timestamp
    planned_exit_ts: pd.Timestamp
    horizon: str
    predicted_gross: float
    predicted_net_zero: float
    predicted_net_kotak: float
    buy_cost_zero: float
    buy_cost_kotak: float


def fit_raw_and_relative_models(labeled: pd.DataFrame, features: list[str], day_start: pd.Timestamp):
    raw_models = {}
    rel_models = {}
    for name, (bars, _) in HORIZONS.items():
        future_ts = f"future_ts_{name}"
        rel_target = f"target_{name}"

        work = labeled.sort_values(["symbol", "timestamp"]).copy()
        g = work.groupby("symbol", sort=False)
        future_close = g["Close"].shift(-bars)
        work[f"raw_{name}"] = future_close / work["Close"] - 1.0

        t = work.dropna(subset=features + [future_ts, rel_target, f"raw_{name}"]).copy()
        t = t[t[future_ts] < day_start]
        if len(t) < 1000:
            raise RuntimeError(f"Too few leakage-safe rows for {name}: {len(t)}")

        params = dict(
            learning_rate=0.035,
            max_iter=260,
            max_leaf_nodes=15,
            min_samples_leaf=70,
            l2_regularization=3.0,
            random_state=42,
        )
        raw = HistGradientBoostingRegressor(**params)
        rel = HistGradientBoostingRegressor(**params)
        raw.fit(t[features], t[f"raw_{name}"])
        rel.fit(t[features], t[rel_target])
        raw_models[name] = raw
        rel_models[name] = rel
        print(
            f"  {name:4} train_rows={len(t):6d} latest_signal={t['timestamp'].max()} "
            f"latest_outcome={t[future_ts].max()}"
        )
    return raw_models, rel_models


def session_bucket(ts_utc: pd.Timestamp):
    local = pd.Timestamp(ts_utc).tz_convert(IST)
    hhmm = local.strftime("%H:%M")
    for name, start, end, quota in SESSION_WINDOWS:
        if start <= hhmm < end:
            return name, quota
    return None, 0


def estimate_round_trip_net(order_value: float, entry_price: float, predicted_gross: float):
    if order_value <= 0 or entry_price <= 0:
        return -1.0, -1.0, 0.0, 0.0
    predicted_sell_value = order_value * (1.0 + predicted_gross)
    buy_zero, buy_kotak = order_costs(order_value, "BUY")
    sell_zero, sell_kotak = order_costs(max(predicted_sell_value, 0.0), "SELL")
    net_zero = predicted_gross - (buy_zero + sell_zero) / order_value
    net_kotak = predicted_gross - (buy_kotak + sell_kotak) / order_value
    return net_zero, net_kotak, buy_zero, buy_kotak


def score_opportunities(raw_models, rel_models, snap: pd.DataFrame, features: list[str], equity: float):
    out = snap[["timestamp", "symbol", "Close"]].copy().reset_index(drop=True)
    rel_combo = np.zeros(len(out), dtype=float)
    agreement = np.zeros(len(out), dtype=int)

    for name, (_, weight) in HORIZONS.items():
        raw_pred = raw_models[name].predict(snap[features])
        rel_pred = rel_models[name].predict(snap[features])
        out[f"raw_{name}"] = raw_pred
        out[f"rel_{name}"] = rel_pred
        out[f"zrel_{name}"] = zscore_cross_section(rel_pred)
        rel_combo += weight * out[f"zrel_{name}"].to_numpy()
        agreement += (raw_pred > 0).astype(int)

    out["relative_score"] = rel_combo
    out["agreement"] = agreement

    # Candidate horizon is the horizon with the highest predicted net return after costs.
    planned_value = max(1.0, equity * CAPITAL_PER_NEW_TRADE)
    best_h = []
    best_gross = []
    best_net_zero = []
    best_net_kotak = []
    for r in out.itertuples(index=False):
        choices = []
        for name in HORIZONS:
            gross = float(getattr(r, f"raw_{name}"))
            nz, nk, _, _ = estimate_round_trip_net(planned_value, float(r.Close), gross)
            choices.append((nk, nz, gross, name))
        choices.sort(reverse=True)
        nk, nz, gross, name = choices[0]
        best_h.append(name)
        best_gross.append(gross)
        best_net_zero.append(nz)
        best_net_kotak.append(nk)

    out["best_horizon"] = best_h
    out["predicted_gross"] = best_gross
    out["predicted_net_zero"] = best_net_zero
    out["predicted_net_kotak"] = best_net_kotak
    out["eligible"] = (
        (out["relative_score"] >= MIN_RELATIVE_SCORE)
        & (out["agreement"] >= MIN_HORIZON_AGREEMENT)
        & (out["predicted_net_kotak"] >= MIN_PREDICTED_NET_RETURN)
    )
    return out.sort_values(["eligible", "predicted_net_kotak", "relative_score"], ascending=[False, False, False])


def horizon_bars(name: str) -> int:
    return HORIZONS[name][0]


def main() -> None:
    print("Share-Trading-AI v23 opportunity-gated intraday replay")
    print(f"Replay date: {REPLAY_DATE} | starting capital: INR {STARTING_CAPITAL:,.0f}")
    print("NO BROKER ORDERS ARE SENT. Historical replay only.")
    print("Scan cadence: every 5 minutes. Trading is OPTIONAL, never mandatory.")
    print(f"Entry gate: predicted KOTAK-profile net return >= {MIN_PREDICTED_NET_RETURN:.2%}")
    print(f"Strength gate: relative_score >= {MIN_RELATIVE_SCORE:.2f} and >= {MIN_HORIZON_AGREEMENT}/5 positive horizons")
    print(f"Daily entry cap={MAX_DAILY_ENTRIES}; session caps OPEN/MID/LATE=10/5/5; max concurrent positions={MAX_POSITIONS}")

    live = build_live_panel(load_events())
    seq, features = feature_block(live)
    labeled = add_horizon_labels(seq)
    day_start, day_end = day_bounds()

    print("\nLeakage-safe training of RAW-return + RELATIVE-strength models:")
    raw_models, rel_models = fit_raw_and_relative_models(labeled, features, day_start)

    friday, times = complete_friday_times(seq, features, day_start, day_end)
    if not times:
        raise RuntimeError("No complete replay bars")
    bars = execution_series(seq)

    cash_zero = STARTING_CAPITAL
    cash_kotak = STARTING_CAPITAL
    positions: dict[str, Position] = {}
    trade_log = []
    rejected = {"weak": 0, "net_gate": 0, "quota": 0, "capacity": 0}
    entries_by_session = {name: 0 for name, *_ in SESSION_WINDOWS}
    total_entries = 0
    equity_curve = []

    for ts in times:
        # Use next-open execution for all actions.
        exec_info = {s: next_open(bars[s], pd.Timestamp(ts)) for s in DEFAULT_CONFIG.universe}
        if any(v is None for v in exec_info.values()):
            continue
        exec_ts = min(v[0] for v in exec_info.values() if v is not None)
        prices = {s: v[1] for s, v in exec_info.items() if v is not None}

        # Exit positions whose selected forecast horizon has matured.
        for symbol in list(positions):
            p = positions[symbol]
            if exec_ts < p.planned_exit_ts:
                continue
            px = prices[symbol]
            value = p.qty * px
            sell_zero, sell_kotak = order_costs(value, "SELL")
            cash_zero += value - sell_zero
            cash_kotak += value - sell_kotak
            gross_pnl = p.qty * (px - p.entry_price)
            net_zero = gross_pnl - p.buy_cost_zero - sell_zero
            net_kotak = gross_pnl - p.buy_cost_kotak - sell_kotak
            trade_log.append({
                "symbol": symbol, "entry_ts": p.entry_ts, "exit_ts": exec_ts,
                "horizon": p.horizon, "qty": p.qty, "entry": p.entry_price, "exit": px,
                "pred_gross": p.predicted_gross, "pred_net_zero": p.predicted_net_zero,
                "pred_net_kotak": p.predicted_net_kotak,
                "actual_gross": px / p.entry_price - 1.0,
                "net_pnl_zero": net_zero, "net_pnl_kotak": net_kotak,
            })
            del positions[symbol]

        equity_zero = cash_zero + sum(p.qty * prices[p.symbol] for p in positions.values())
        equity_kotak = cash_kotak + sum(p.qty * prices[p.symbol] for p in positions.values())

        snap = friday[friday["timestamp"] == ts].copy().sort_values("symbol")
        if snap["symbol"].nunique() != len(DEFAULT_CONFIG.universe):
            continue
        scored = score_opportunities(raw_models, rel_models, snap, features, equity_kotak)

        session, session_cap = session_bucket(exec_ts)
        eligible = scored[scored["eligible"]].copy()
        if eligible.empty:
            # Record why the strongest opportunity failed.
            top = scored.iloc[0]
            if top["relative_score"] < MIN_RELATIVE_SCORE or int(top["agreement"]) < MIN_HORIZON_AGREEMENT:
                rejected["weak"] += 1
            elif top["predicted_net_kotak"] < MIN_PREDICTED_NET_RETURN:
                rejected["net_gate"] += 1
        else:
            for r in eligible.itertuples(index=False):
                symbol = str(r.symbol)
                if symbol in positions:
                    continue
                if len(positions) >= MAX_POSITIONS:
                    rejected["capacity"] += 1
                    break
                if total_entries >= MAX_DAILY_ENTRIES or session is None or entries_by_session[session] >= session_cap:
                    rejected["quota"] += 1
                    break

                # Allocate at most 30% of equity, while preserving a cash buffer.
                target_value = equity_kotak * CAPITAL_PER_NEW_TRADE
                max_spend = max(0.0, cash_kotak - equity_kotak * CASH_BUFFER)
                order_value = min(target_value, max_spend)
                px = prices[symbol]
                qty = int(order_value // px)
                if qty <= 0:
                    continue
                order_value = qty * px

                # Recompute the candidate's cost-adjusted expected return using actual order size.
                pred_gross = float(r.predicted_gross)
                nz, nk, buy_zero, buy_kotak = estimate_round_trip_net(order_value, px, pred_gross)
                if nk < MIN_PREDICTED_NET_RETURN:
                    rejected["net_gate"] += 1
                    continue
                if order_value + buy_zero > cash_zero or order_value + buy_kotak > cash_kotak:
                    continue

                h = str(r.best_horizon)
                planned_exit = exec_ts + pd.Timedelta(minutes=5 * horizon_bars(h))
                # No new trade whose planned horizon extends beyond normal intraday square-off.
                if planned_exit.tz_convert(IST).strftime("%H:%M") > "15:20":
                    continue

                cash_zero -= order_value + buy_zero
                cash_kotak -= order_value + buy_kotak
                positions[symbol] = Position(
                    symbol=symbol, qty=qty, entry_price=px, entry_ts=exec_ts,
                    planned_exit_ts=planned_exit, horizon=h,
                    predicted_gross=pred_gross, predicted_net_zero=nz,
                    predicted_net_kotak=nk, buy_cost_zero=buy_zero, buy_cost_kotak=buy_kotak,
                )
                entries_by_session[session] += 1
                total_entries += 1

        equity_zero = cash_zero + sum(p.qty * prices[p.symbol] for p in positions.values())
        equity_kotak = cash_kotak + sum(p.qty * prices[p.symbol] for p in positions.values())
        equity_curve.append((exec_ts, equity_zero, equity_kotak, len(positions)))

    # Force intraday square-off at last available open.
    for symbol in list(positions):
        p = positions[symbol]
        s = bars[symbol]
        day_rows = s[(s["timestamp"] >= day_start) & (s["timestamp"] < day_end)]
        if day_rows.empty:
            continue
        rr = day_rows.iloc[-1]
        px = float(rr["Open"])
        exit_ts = pd.Timestamp(rr["timestamp"])
        value = p.qty * px
        sell_zero, sell_kotak = order_costs(value, "SELL")
        cash_zero += value - sell_zero
        cash_kotak += value - sell_kotak
        gross_pnl = p.qty * (px - p.entry_price)
        trade_log.append({
            "symbol": symbol, "entry_ts": p.entry_ts, "exit_ts": exit_ts,
            "horizon": p.horizon, "qty": p.qty, "entry": p.entry_price, "exit": px,
            "pred_gross": p.predicted_gross, "pred_net_zero": p.predicted_net_zero,
            "pred_net_kotak": p.predicted_net_kotak,
            "actual_gross": px / p.entry_price - 1.0,
            "net_pnl_zero": gross_pnl - p.buy_cost_zero - sell_zero,
            "net_pnl_kotak": gross_pnl - p.buy_cost_kotak - sell_kotak,
        })
        del positions[symbol]

    trades = pd.DataFrame(trade_log)
    print("\nV23 EXECUTED TRADES")
    if trades.empty:
        print("  No trade passed the opportunity + predicted-net-profit gates. CASH was preserved.")
    else:
        for i, r in enumerate(trades.itertuples(index=False), start=1):
            print(
                f"  {i:2d} {r.symbol:13} {r.horizon:4} BUY={r.entry_ts} @{r.entry:.2f} "
                f"SELL={r.exit_ts} @{r.exit:.2f} qty={r.qty} "
                f"pred_net_kotak={r.pred_net_kotak:+.2%} actual_gross={r.actual_gross:+.2%} "
                f"PnL_zero=INR {r.net_pnl_zero:+,.2f} PnL_kotak=INR {r.net_pnl_kotak:+,.2f}"
            )

    print("\nV23 OPPORTUNITY-GATED SUMMARY")
    print(f"  scans performed: {len(equity_curve)}")
    print(f"  entries executed: {total_entries} / max {MAX_DAILY_ENTRIES}")
    print(f"  session entries: {entries_by_session}")
    print(f"  rejected/idle scans: {rejected}")
    print(f"  completed trades: {len(trades)}")
    if not trades.empty:
        print(f"  predicted-net gate: >= {MIN_PREDICTED_NET_RETURN:.2%} under Kotak profile")
        print(f"  actual profitable trades ZERO: {(trades['net_pnl_zero'] > 0).mean():.2%}")
        print(f"  actual profitable trades KOTAK: {(trades['net_pnl_kotak'] > 0).mean():.2%}")
        print(f"  avg predicted net KOTAK: {trades['pred_net_kotak'].mean():+.3%}")
        print(f"  avg actual gross return: {trades['actual_gross'].mean():+.3%}")
    print(f"  ZERO-brokerage final capital: INR {cash_zero:,.2f} | return={cash_zero/STARTING_CAPITAL-1:+.3%}")
    print(f"  KOTAK-profile final capital: INR {cash_kotak:,.2f} | return={cash_kotak/STARTING_CAPITAL-1:+.3%}")
    print("\nInterpretation: scanning is continuous, but execution is conditional. A zero-trade day is a valid outcome.")
    print("Research only. This is not live-trading approval.")


if __name__ == "__main__":
    main()
