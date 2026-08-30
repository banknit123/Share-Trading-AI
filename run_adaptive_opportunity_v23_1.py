from __future__ import annotations

import math
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
    STARTING_CAPITAL,
    MAX_POSITIONS,
    add_horizon_labels,
    complete_friday_times,
    execution_series,
    next_open,
    order_costs,
    zscore_cross_section,
)

REPLAY_DATE = "2026-08-28"
CALIBRATION_SESSIONS = 10
CAPITAL_PER_NEW_TRADE = 0.30
CASH_BUFFER = 0.10

SESSION_WINDOWS = [
    ("OPEN", "09:15", "11:15"),
    ("MID",  "11:15", "13:15"),
    ("LATE", "13:15", "15:15"),
]


@dataclass
class LearnedGates:
    min_predicted_net: float
    min_relative_score: float
    min_agreement: int
    max_daily_entries: int
    session_caps: dict[str, int]
    calibration_rows: int
    calibration_win_rate: float
    calibration_mean_actual_net: float


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


def day_bounds(date_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(date_str, tz=IST)
    return start.tz_convert("UTC"), (start + pd.Timedelta(days=1)).tz_convert("UTC")


def fit_raw_and_relative_models(labeled: pd.DataFrame, features: list[str], outcome_cutoff: pd.Timestamp):
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
        t = t[t[future_ts] < outcome_cutoff]
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
    return raw_models, rel_models


def session_bucket(ts_utc: pd.Timestamp) -> str | None:
    local = pd.Timestamp(ts_utc).tz_convert(IST)
    hhmm = local.strftime("%H:%M")
    for name, start, end in SESSION_WINDOWS:
        if start <= hhmm < end:
            return name
    return None


def estimate_round_trip_net(order_value: float, predicted_gross: float):
    if order_value <= 0:
        return -1.0, -1.0, 0.0, 0.0
    sell_value = max(0.0, order_value * (1.0 + predicted_gross))
    buy_zero, buy_kotak = order_costs(order_value, "BUY")
    sell_zero, sell_kotak = order_costs(sell_value, "SELL")
    return (
        predicted_gross - (buy_zero + sell_zero) / order_value,
        predicted_gross - (buy_kotak + sell_kotak) / order_value,
        buy_zero,
        buy_kotak,
    )


def score_opportunities(raw_models, rel_models, snap: pd.DataFrame, features: list[str], equity: float):
    out = snap[["timestamp", "symbol", "Close"]].copy().reset_index(drop=True)
    rel_combo = np.zeros(len(out), dtype=float)
    agreement = np.zeros(len(out), dtype=int)
    planned_value = max(1.0, equity * CAPITAL_PER_NEW_TRADE)

    for name, (_, weight) in HORIZONS.items():
        rp = raw_models[name].predict(snap[features])
        xp = rel_models[name].predict(snap[features])
        out[f"raw_{name}"] = rp
        out[f"rel_{name}"] = xp
        rel_combo += weight * zscore_cross_section(xp)
        agreement += (rp > 0).astype(int)

    out["relative_score"] = rel_combo
    out["agreement"] = agreement

    best_h, best_gross, best_net_zero, best_net_kotak = [], [], [], []
    for r in out.itertuples(index=False):
        choices = []
        for name in HORIZONS:
            gross = float(getattr(r, f"raw_{name}"))
            nz, nk, _, _ = estimate_round_trip_net(planned_value, gross)
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
    return out


def actual_net_for_candidate(row, bars: dict[str, pd.DataFrame], order_value: float):
    symbol = str(row.symbol)
    series = bars[symbol]
    idx = series.index[series["timestamp"] == pd.Timestamp(row.timestamp)].to_numpy()
    if len(idx) == 0:
        return None
    entry_pos = int(idx[0]) + 1
    hbars = HORIZONS[str(row.best_horizon)][0]
    exit_pos = entry_pos + hbars
    if entry_pos >= len(series) or exit_pos >= len(series):
        return None
    entry = series.iloc[entry_pos]
    exit_ = series.iloc[exit_pos]
    # Require same trading day for intraday calibration.
    if pd.Timestamp(entry["timestamp"]).tz_convert(IST).date() != pd.Timestamp(exit_["timestamp"]).tz_convert(IST).date():
        return None
    entry_px = float(entry["Open"])
    exit_px = float(exit_["Open"])
    gross = exit_px / entry_px - 1.0
    _, net_kotak, _, _ = estimate_round_trip_net(order_value, gross)
    return gross, net_kotak


def calibration_dates(seq: pd.DataFrame, replay_start: pd.Timestamp) -> list[str]:
    work = seq[seq["timestamp"] < replay_start].copy()
    local_dates = pd.Series(work["timestamp"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d").unique()).sort_values()
    return list(local_dates.tail(CALIBRATION_SESSIONS))


def build_calibration_table(seq: pd.DataFrame, features: list[str], labeled: pd.DataFrame, replay_start: pd.Timestamp, bars):
    dates = calibration_dates(seq, replay_start)
    if len(dates) < 5:
        raise RuntimeError("Too few historical sessions for calibration")
    cal_start, _ = day_bounds(dates[0])
    raw_models, rel_models = fit_raw_and_relative_models(labeled, features, cal_start)
    rows = []
    order_value = STARTING_CAPITAL * CAPITAL_PER_NEW_TRADE

    for date_str in dates:
        ds, de = day_bounds(date_str)
        day = seq[(seq["timestamp"] >= ds) & (seq["timestamp"] < de)].dropna(subset=features).copy()
        counts = day.groupby("timestamp")["symbol"].nunique()
        times = list(counts[counts == len(DEFAULT_CONFIG.universe)].index)
        for ts in times:
            snap = day[day["timestamp"] == ts].copy().sort_values("symbol")
            scored = score_opportunities(raw_models, rel_models, snap, features, STARTING_CAPITAL)
            for r in scored.itertuples(index=False):
                actual = actual_net_for_candidate(r, bars, order_value)
                if actual is None:
                    continue
                gross, actual_net = actual
                rows.append({
                    "date": date_str,
                    "timestamp": pd.Timestamp(ts),
                    "session": session_bucket(pd.Timestamp(ts)),
                    "symbol": str(r.symbol),
                    "predicted_net": float(r.predicted_net_kotak),
                    "relative_score": float(r.relative_score),
                    "agreement": int(r.agreement),
                    "actual_gross": gross,
                    "actual_net": actual_net,
                })
    table = pd.DataFrame(rows)
    if len(table) < 200:
        raise RuntimeError(f"Calibration table too small: {len(table)}")
    return table, dates


def learn_gates(cal: pd.DataFrame) -> LearnedGates:
    # Search only a compact, pre-declared grid. Selection objective rewards actual net return,
    # breadth across days, and win rate while penalising excessive turnover.
    pred_qs = [0.70, 0.80, 0.85, 0.90, 0.95]
    rel_qs = [0.50, 0.65, 0.75, 0.85]
    agree_vals = [2, 3, 4, 5]
    candidates = []

    for pq in pred_qs:
        pth = float(cal["predicted_net"].quantile(pq))
        for rq in rel_qs:
            rth = float(cal["relative_score"].quantile(rq))
            for agree in agree_vals:
                x = cal[(cal["predicted_net"] >= pth) & (cal["relative_score"] >= rth) & (cal["agreement"] >= agree)].copy()
                if len(x) < 25:
                    continue
                by_day = x.groupby("date")["actual_net"].mean()
                covered_days = int((x.groupby("date").size() > 0).sum())
                if covered_days < 5:
                    continue
                mean_net = float(x["actual_net"].mean())
                win = float((x["actual_net"] > 0).mean())
                median_day_net = float(by_day.median())
                avg_per_day = float(x.groupby("date").size().mean())
                # Prefer positive out-of-sample calibration economics and reasonable turnover.
                objective = mean_net + 0.5 * median_day_net + 0.002 * (win - 0.5) - 0.00002 * max(avg_per_day - 20.0, 0.0)
                candidates.append((objective, pth, rth, agree, x, mean_net, win, avg_per_day))

    if not candidates:
        raise RuntimeError("No calibration gate had enough observations")
    candidates.sort(key=lambda z: z[0], reverse=True)
    _, pth, rth, agree, x, mean_net, win, avg_per_day = candidates[0]

    # Daily max derives from the historical 75th percentile of qualifying opportunity counts.
    daily_counts = x.groupby("date").size()
    max_daily = int(np.clip(math.ceil(float(daily_counts.quantile(0.75))), 1, 30))

    # Session capacity is proportional to historically profitable qualifying opportunities.
    profit_counts = x[x["actual_net"] > 0].groupby("session").size().reindex(["OPEN", "MID", "LATE"], fill_value=0)
    if int(profit_counts.sum()) == 0:
        session_caps = {"OPEN": max(1, max_daily // 3), "MID": max(1, max_daily // 3), "LATE": max(1, max_daily - 2 * max(1, max_daily // 3))}
    else:
        shares = profit_counts / profit_counts.sum()
        raw_caps = {s: max(1, int(round(max_daily * float(shares[s])))) for s in shares.index}
        # Reconcile to total max_daily while preserving at least one slot per window.
        while sum(raw_caps.values()) > max_daily:
            k = max(raw_caps, key=raw_caps.get)
            if raw_caps[k] > 1:
                raw_caps[k] -= 1
            else:
                break
        while sum(raw_caps.values()) < max_daily:
            k = max(shares.index, key=lambda s: float(shares[s]))
            raw_caps[k] += 1
        session_caps = raw_caps

    return LearnedGates(
        min_predicted_net=max(0.0, pth),
        min_relative_score=rth,
        min_agreement=agree,
        max_daily_entries=max_daily,
        session_caps=session_caps,
        calibration_rows=len(x),
        calibration_win_rate=win,
        calibration_mean_actual_net=mean_net,
    )


def main() -> None:
    print("Share-Trading-AI v23.1 adaptive opportunity-gated replay")
    print("NO BROKER ORDERS ARE SENT. Historical replay only.")
    print("Manual 1% and 10/5/5 examples are REMOVED; gates are learned from prior stock behaviour only.")

    live = build_live_panel(load_events())
    seq, features = feature_block(live)
    labeled = add_horizon_labels(seq)
    bars = execution_series(seq)
    replay_start, replay_end = day_bounds(REPLAY_DATE)

    print("\nBuilding pre-Friday calibration study...")
    cal, cal_dates = build_calibration_table(seq, features, labeled, replay_start, bars)
    gates = learn_gates(cal)
    print(f"Calibration sessions: {cal_dates[0]} -> {cal_dates[-1]} ({len(cal_dates)} sessions)")
    print(f"Candidate observations studied: {len(cal)}")
    print("\nLEARNED GATES")
    print(f"  minimum predicted net return: {gates.min_predicted_net:+.3%}")
    print(f"  minimum relative score: {gates.min_relative_score:+.3f}")
    print(f"  minimum positive horizons: {gates.min_agreement}/5")
    print(f"  learned daily entry cap: {gates.max_daily_entries}")
    print(f"  learned session caps: {gates.session_caps}")
    print(f"  calibration qualifying rows: {gates.calibration_rows}")
    print(f"  calibration win rate: {gates.calibration_win_rate:.2%}")
    print(f"  calibration mean actual net/trade: {gates.calibration_mean_actual_net:+.3%}")

    # Retrain models on all outcomes known before Friday.
    raw_models, rel_models = fit_raw_and_relative_models(labeled, features, replay_start)
    friday, times = complete_friday_times(seq, features, replay_start, replay_end)
    if not times:
        raise RuntimeError("No complete Friday replay bars")

    cash_zero = STARTING_CAPITAL
    cash_kotak = STARTING_CAPITAL
    positions: dict[str, Position] = {}
    trade_log = []
    entries_by_session = {"OPEN": 0, "MID": 0, "LATE": 0}
    total_entries = 0
    idle_scans = 0

    for ts in times:
        exec_info = {s: next_open(bars[s], pd.Timestamp(ts)) for s in DEFAULT_CONFIG.universe}
        if any(v is None for v in exec_info.values()):
            continue
        exec_ts = min(v[0] for v in exec_info.values() if v is not None)
        prices = {s: v[1] for s, v in exec_info.items() if v is not None}

        # Planned exits.
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
            trade_log.append({
                "symbol": symbol, "entry_ts": p.entry_ts, "exit_ts": exec_ts,
                "horizon": p.horizon, "qty": p.qty, "entry": p.entry_price, "exit": px,
                "pred_net_kotak": p.predicted_net_kotak,
                "actual_gross": px / p.entry_price - 1.0,
                "net_pnl_zero": gross_pnl - p.buy_cost_zero - sell_zero,
                "net_pnl_kotak": gross_pnl - p.buy_cost_kotak - sell_kotak,
            })
            del positions[symbol]

        snap = friday[friday["timestamp"] == ts].copy().sort_values("symbol")
        if snap["symbol"].nunique() != len(DEFAULT_CONFIG.universe):
            continue
        equity_kotak = cash_kotak + sum(p.qty * prices[p.symbol] for p in positions.values())
        scored = score_opportunities(raw_models, rel_models, snap, features, equity_kotak)
        eligible = scored[
            (scored["predicted_net_kotak"] >= gates.min_predicted_net)
            & (scored["relative_score"] >= gates.min_relative_score)
            & (scored["agreement"] >= gates.min_agreement)
        ].sort_values(["predicted_net_kotak", "relative_score"], ascending=False)

        session = session_bucket(exec_ts)
        placed = 0
        for r in eligible.itertuples(index=False):
            symbol = str(r.symbol)
            if symbol in positions or len(positions) >= MAX_POSITIONS:
                continue
            if total_entries >= gates.max_daily_entries:
                break
            if session is None or entries_by_session[session] >= gates.session_caps.get(session, 0):
                break

            target_value = equity_kotak * CAPITAL_PER_NEW_TRADE
            max_spend = max(0.0, cash_kotak - equity_kotak * CASH_BUFFER)
            order_value = min(target_value, max_spend)
            px = prices[symbol]
            qty = int(order_value // px)
            if qty <= 0:
                continue
            order_value = qty * px
            pred_gross = float(r.predicted_gross)
            nz, nk, buy_zero, buy_kotak = estimate_round_trip_net(order_value, pred_gross)
            if nk < gates.min_predicted_net:
                continue
            if order_value + buy_zero > cash_zero or order_value + buy_kotak > cash_kotak:
                continue
            h = str(r.best_horizon)
            planned_exit = exec_ts + pd.Timedelta(minutes=5 * HORIZONS[h][0])
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
            total_entries += 1
            entries_by_session[session] += 1
            placed += 1
        if placed == 0:
            idle_scans += 1

    # Intraday square-off.
    for symbol in list(positions):
        p = positions[symbol]
        s = bars[symbol]
        day_rows = s[(s["timestamp"] >= replay_start) & (s["timestamp"] < replay_end)]
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
            "pred_net_kotak": p.predicted_net_kotak,
            "actual_gross": px / p.entry_price - 1.0,
            "net_pnl_zero": gross_pnl - p.buy_cost_zero - sell_zero,
            "net_pnl_kotak": gross_pnl - p.buy_cost_kotak - sell_kotak,
        })
        del positions[symbol]

    trades = pd.DataFrame(trade_log)
    print("\nV23.1 EXECUTED TRADES")
    if trades.empty:
        print("  No Friday opportunity passed the data-learned gates. Cash preserved.")
    else:
        for i, r in enumerate(trades.itertuples(index=False), start=1):
            print(
                f"  {i:2d} {r.symbol:13} {r.horizon:4} BUY={r.entry_ts} @{r.entry:.2f} "
                f"SELL={r.exit_ts} @{r.exit:.2f} qty={r.qty} pred_net={r.pred_net_kotak:+.2%} "
                f"actual_gross={r.actual_gross:+.2%} PnL_zero=INR {r.net_pnl_zero:+.2f} PnL_kotak=INR {r.net_pnl_kotak:+.2f}"
            )

    print("\nV23.1 ADAPTIVE SUMMARY")
    print(f"  entries executed: {total_entries} / learned daily cap {gates.max_daily_entries}")
    print(f"  session entries: {entries_by_session} / learned caps {gates.session_caps}")
    print(f"  idle scans: {idle_scans}")
    print(f"  completed trades: {len(trades)}")
    if not trades.empty:
        print(f"  actual profitable trades (Kotak costs): {(trades['net_pnl_kotak'] > 0).mean():.2%}")
        print(f"  avg predicted net: {trades['pred_net_kotak'].mean():+.3%}")
        print(f"  avg actual gross return: {trades['actual_gross'].mean():+.3%}")
    print(f"  ZERO-brokerage final capital: INR {cash_zero:,.2f} | return={cash_zero/STARTING_CAPITAL-1:+.3%}")
    print(f"  KOTAK-profile final capital: INR {cash_kotak:,.2f} | return={cash_kotak/STARTING_CAPITAL-1:+.3%}")
    print("\nResearch only. The learned gates are based on pre-Friday calibration data, not on Friday outcomes.")


if __name__ == "__main__":
    main()
