from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_ai.config import DEFAULT_CONFIG
from run_forward_paper_observer_v19 import build_live_panel, feature_block
from run_historical_multimodal_v10 import load_events
from run_intraday_portfolio_v21 import HORIZONS, IST, STARTING_CAPITAL, MAX_POSITIONS, add_horizon_labels, complete_friday_times, execution_series, next_open, order_costs
import run_adaptive_opportunity_v23_1 as v231

REPLAY_DATE = "2026-08-28"
CALIBRATION_SESSIONS = 20
CAPITAL_PER_NEW_TRADE = 0.30
CASH_BUFFER = 0.10


@dataclass
class RobustGates:
    active: bool
    min_predicted_net: float = np.nan
    min_relative_score: float = np.nan
    min_agreement: int = 5
    max_daily_entries: int = 0
    session_caps: dict[str, int] | None = None
    train_rows: int = 0
    validation_rows: int = 0
    train_mean_net: float = np.nan
    validation_mean_net: float = np.nan
    train_win_rate: float = np.nan
    validation_win_rate: float = np.nan
    train_positive_day_rate: float = np.nan
    validation_positive_day_rate: float = np.nan
    conservative_edge: float = np.nan
    reason: str = ""


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


def stats(x: pd.DataFrame) -> dict:
    by_day = x.groupby("date")["actual_net"].mean()
    return {
        "n": int(len(x)),
        "days": int(x["date"].nunique()),
        "mean": float(x["actual_net"].mean()),
        "median": float(x["actual_net"].median()),
        "win": float((x["actual_net"] > 0).mean()),
        "median_day": float(by_day.median()),
        "positive_day_rate": float((by_day > 0).mean()),
        "se": float(x["actual_net"].std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.inf,
    }


def learn_robust_gates(cal: pd.DataFrame) -> RobustGates:
    dates = sorted(cal["date"].dropna().unique())
    if len(dates) < 10:
        return RobustGates(active=False, reason="Fewer than 10 calibration sessions")

    split = max(5, int(np.floor(len(dates) * 0.60)))
    split = min(split, len(dates) - 4)
    train_dates = dates[:split]
    val_dates = dates[split:]
    gate_train = cal[cal["date"].isin(train_dates)].copy()
    gate_val = cal[cal["date"].isin(val_dates)].copy()

    # Threshold values are derived ONLY from the earlier gate-training block.
    pred_qs = [0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975]
    rel_qs = [0.50, 0.65, 0.75, 0.85, 0.90]
    agreements = [3, 4, 5]
    viable = []

    for pq in pred_qs:
        pth = float(gate_train["predicted_net"].quantile(pq))
        for rq in rel_qs:
            rth = float(gate_train["relative_score"].quantile(rq))
            for agree in agreements:
                tr = gate_train[(gate_train["predicted_net"] >= pth) & (gate_train["relative_score"] >= rth) & (gate_train["agreement"] >= agree)].copy()
                va = gate_val[(gate_val["predicted_net"] >= pth) & (gate_val["relative_score"] >= rth) & (gate_val["agreement"] >= agree)].copy()
                if len(tr) < 25 or len(va) < 12:
                    continue
                if tr["date"].nunique() < max(4, len(train_dates) // 2):
                    continue
                if va["date"].nunique() < max(3, len(val_dates) // 2):
                    continue

                st = stats(tr)
                sv = stats(va)

                # A gate is tradeable only if economics are positive in BOTH blocks.
                # This prevents a threshold with negative expected value from being chosen.
                if st["mean"] <= 0 or sv["mean"] <= 0:
                    continue
                if st["median_day"] <= 0 or sv["median_day"] <= 0:
                    continue
                if st["win"] <= 0.50 or sv["win"] <= 0.50:
                    continue
                if st["positive_day_rate"] < 0.55 or sv["positive_day_rate"] < 0.60:
                    continue

                conservative_edge = min(st["mean"], sv["mean"])
                conservative_lcb = min(st["mean"] - 0.50 * st["se"], sv["mean"] - 0.50 * sv["se"])
                if conservative_lcb <= 0:
                    continue

                # Prefer repeatable net edge, then stability, then lower turnover.
                avg_daily = float(pd.concat([tr, va]).groupby("date").size().mean())
                objective = (
                    conservative_edge
                    + 0.50 * conservative_lcb
                    + 0.20 * min(st["median_day"], sv["median_day"])
                    + 0.001 * (min(st["win"], sv["win"]) - 0.50)
                    - 0.00001 * max(avg_daily - 10.0, 0.0)
                )
                viable.append((objective, pth, rth, agree, tr, va, st, sv, conservative_edge))

    if not viable:
        return RobustGates(
            active=False,
            reason="No threshold produced positive, stable after-cost expectancy in both gate-training and later validation blocks",
        )

    viable.sort(key=lambda z: z[0], reverse=True)
    _, pth, rth, agree, tr, va, st, sv, edge = viable[0]

    selected = pd.concat([tr, va], ignore_index=True)
    # Daily trade capacity comes from observed qualifying opportunity frequency, not a manual cap.
    daily_counts = selected.groupby("date").size()
    max_daily = max(1, int(np.ceil(float(daily_counts.quantile(0.75)))))

    # Reserve capacity only for sessions that showed positive validation expectancy.
    session_caps: dict[str, int] = {"OPEN": 0, "MID": 0, "LATE": 0}
    val_session_stats = {}
    for session in session_caps:
        s = va[va["session"] == session]
        if len(s) >= 5 and float(s["actual_net"].mean()) > 0:
            val_session_stats[session] = s

    if not val_session_stats:
        return RobustGates(
            active=False,
            reason="A robust global gate existed, but no session window retained positive validation expectancy",
        )

    raw_caps = {}
    for session, s in val_session_stats.items():
        per_day = s.groupby("date").size()
        raw_caps[session] = max(1, int(np.ceil(float(per_day.quantile(0.75)))))

    # If raw session capacities exceed the learned daily opportunity budget, scale them proportionally.
    total_raw = sum(raw_caps.values())
    if total_raw <= max_daily:
        session_caps.update(raw_caps)
    else:
        shares = {k: v / total_raw for k, v in raw_caps.items()}
        for k in raw_caps:
            session_caps[k] = max(1, int(np.floor(max_daily * shares[k])))
        while sum(session_caps.values()) < max_daily:
            best = max(raw_caps, key=lambda k: shares[k])
            session_caps[best] += 1
        while sum(session_caps.values()) > max_daily:
            candidates = [k for k, v in session_caps.items() if v > 1]
            if not candidates:
                break
            worst = min(candidates, key=lambda k: shares.get(k, 0.0))
            session_caps[worst] -= 1

    return RobustGates(
        active=True,
        min_predicted_net=max(0.0, pth),
        min_relative_score=rth,
        min_agreement=agree,
        max_daily_entries=max_daily,
        session_caps=session_caps,
        train_rows=st["n"],
        validation_rows=sv["n"],
        train_mean_net=st["mean"],
        validation_mean_net=sv["mean"],
        train_win_rate=st["win"],
        validation_win_rate=sv["win"],
        train_positive_day_rate=st["positive_day_rate"],
        validation_positive_day_rate=sv["positive_day_rate"],
        conservative_edge=edge,
        reason="Robust gate passed",
    )


def main() -> None:
    print("Share-Trading-AI v24 robust opportunity / abstention replay")
    print("NO BROKER ORDERS ARE SENT. Historical replay only.")
    print("Principle: scan every 5m, but trade only if a gate is profitable across TWO prior time blocks after costs.")
    print("If no robust gate exists, the correct decision is CASH / NO TRADE.")

    live = build_live_panel(load_events())
    seq, features = feature_block(live)
    labeled = add_horizon_labels(seq)
    bars = execution_series(seq)
    replay_start, replay_end = v231.day_bounds(REPLAY_DATE)

    # Expand the pre-Friday study if data is available. v23.1 functions reference this module global.
    v231.CALIBRATION_SESSIONS = CALIBRATION_SESSIONS
    print("\nBuilding robust pre-Friday opportunity study...")
    cal, cal_dates = v231.build_calibration_table(seq, features, labeled, replay_start, bars)
    gates = learn_robust_gates(cal)

    print(f"Calibration sessions: {cal_dates[0]} -> {cal_dates[-1]} ({len(cal_dates)} sessions)")
    print(f"Candidate observations studied: {len(cal)}")
    print("\nV24 ROBUST GATE")
    print(f"  active: {gates.active}")
    print(f"  reason: {gates.reason}")

    if not gates.active:
        print("\nV24 FRIDAY RESULT")
        print("  NO TRADE. Capital preserved because historical evidence did not justify a positive-expectancy entry gate.")
        print(f"  starting capital: INR {STARTING_CAPITAL:,.2f}")
        print(f"  final capital:    INR {STARTING_CAPITAL:,.2f} | return=+0.000%")
        return

    print(f"  minimum predicted net return: {gates.min_predicted_net:+.3%}")
    print(f"  minimum relative score: {gates.min_relative_score:+.3f}")
    print(f"  minimum positive horizons: {gates.min_agreement}/5")
    print(f"  learned daily entry cap: {gates.max_daily_entries}")
    print(f"  learned session caps: {gates.session_caps}")
    print(f"  gate-train rows: {gates.train_rows} mean_net={gates.train_mean_net:+.3%} win={gates.train_win_rate:.2%} positive_days={gates.train_positive_day_rate:.2%}")
    print(f"  later-validation rows: {gates.validation_rows} mean_net={gates.validation_mean_net:+.3%} win={gates.validation_win_rate:.2%} positive_days={gates.validation_positive_day_rate:.2%}")
    print(f"  conservative historical net edge/trade: {gates.conservative_edge:+.3%}")

    raw_models, rel_models = v231.fit_raw_and_relative_models(labeled, features, replay_start)
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
        scored = v231.score_opportunities(raw_models, rel_models, snap, features, equity_kotak)
        eligible = scored[
            (scored["predicted_net_kotak"] >= gates.min_predicted_net)
            & (scored["relative_score"] >= gates.min_relative_score)
            & (scored["agreement"] >= gates.min_agreement)
        ].sort_values(["predicted_net_kotak", "relative_score"], ascending=False)

        session = v231.session_bucket(exec_ts)
        placed = 0
        for r in eligible.itertuples(index=False):
            symbol = str(r.symbol)
            if symbol in positions or len(positions) >= MAX_POSITIONS:
                continue
            if total_entries >= gates.max_daily_entries:
                break
            if session is None or gates.session_caps is None or entries_by_session[session] >= gates.session_caps.get(session, 0):
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
            nz, nk, buy_zero, buy_kotak = v231.estimate_round_trip_net(order_value, pred_gross)
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

    # Force intraday square-off.
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
    print("\nV24 EXECUTED TRADES")
    if trades.empty:
        print("  No Friday opportunity passed the robust positive-expectancy gate. CASH preserved.")
    else:
        for i, r in enumerate(trades.itertuples(index=False), start=1):
            print(
                f"  {i:2d} {r.symbol:13} {r.horizon:4} BUY={r.entry_ts} @{r.entry:.2f} "
                f"SELL={r.exit_ts} @{r.exit:.2f} qty={r.qty} pred_net={r.pred_net_kotak:+.2%} "
                f"actual_gross={r.actual_gross:+.2%} PnL_zero=INR {r.net_pnl_zero:+.2f} PnL_kotak=INR {r.net_pnl_kotak:+.2f}"
            )

    print("\nV24 ROBUST SUMMARY")
    print(f"  entries executed: {total_entries} / learned daily cap {gates.max_daily_entries}")
    print(f"  session entries: {entries_by_session} / learned caps {gates.session_caps}")
    print(f"  idle scans: {idle_scans}")
    print(f"  completed trades: {len(trades)}")
    if not trades.empty:
        print(f"  profitable trades after Kotak costs: {(trades['net_pnl_kotak'] > 0).mean():.2%}")
        print(f"  avg predicted net: {trades['pred_net_kotak'].mean():+.3%}")
        print(f"  avg actual gross return: {trades['actual_gross'].mean():+.3%}")
        print(f"  total net PnL ZERO: INR {trades['net_pnl_zero'].sum():+,.2f}")
        print(f"  total net PnL KOTAK: INR {trades['net_pnl_kotak'].sum():+,.2f}")
    print(f"  ZERO-brokerage final capital: INR {cash_zero:,.2f} | return={cash_zero/STARTING_CAPITAL-1:+.3%}")
    print(f"  KOTAK-profile final capital: INR {cash_kotak:,.2f} | return={cash_kotak/STARTING_CAPITAL-1:+.3%}")
    print("\nResearch only. v24 is allowed to abstain completely when prior after-cost evidence is not robust.")


if __name__ == "__main__":
    main()
