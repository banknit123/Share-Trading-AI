from __future__ import annotations

import numpy as np
import pandas as pd

from trading_ai.config import DEFAULT_CONFIG
from run_forward_paper_observer_v19 import build_live_panel, feature_block
from run_historical_multimodal_v10 import load_events
from run_intraday_portfolio_v21 import (
    HORIZONS,
    IST,
    STARTING_CAPITAL,
    MAX_POSITIONS,
    add_horizon_labels,
    execution_series,
    next_open,
    order_costs,
)
import run_adaptive_opportunity_v23_1 as v231
from run_robust_opportunity_v24 import learn_robust_gates, Position

TEST_SESSIONS = 20
CALIBRATION_SESSIONS = 20
CAPITAL_PER_NEW_TRADE = 0.30
CASH_BUFFER = 0.10


def available_dates(seq: pd.DataFrame) -> list[str]:
    work = seq.copy()
    work["date"] = work["timestamp"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d")
    counts = work.groupby(["date", "timestamp"])["symbol"].nunique().reset_index()
    full = counts[counts["symbol"] == len(DEFAULT_CONFIG.universe)]
    by_date = full.groupby("date")["timestamp"].nunique()
    # Require a meaningful intraday session rather than a fragment.
    return sorted(by_date[by_date >= 40].index.tolist())


def replay_one_day(
    date_str: str,
    starting_capital: float,
    seq: pd.DataFrame,
    features: list[str],
    labeled: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
) -> dict:
    replay_start, replay_end = v231.day_bounds(date_str)

    # Build robust gate only from sessions strictly before the replay date.
    v231.CALIBRATION_SESSIONS = CALIBRATION_SESSIONS
    try:
        cal, cal_dates = v231.build_calibration_table(seq, features, labeled, replay_start, bars)
    except Exception as exc:
        return {
            "date": date_str,
            "gate_active": False,
            "reason": f"calibration unavailable: {exc}",
            "cal_sessions": 0,
            "trades": 0,
            "wins": 0,
            "gross_return": 0.0,
            "zero_return": 0.0,
            "kotak_return": 0.0,
            "ending_zero": starting_capital,
            "ending_kotak": starting_capital,
            "predicted_positive": 0,
            "direction_correct": 0,
        }

    gates = learn_robust_gates(cal)
    if not gates.active:
        return {
            "date": date_str,
            "gate_active": False,
            "reason": gates.reason,
            "cal_sessions": len(cal_dates),
            "trades": 0,
            "wins": 0,
            "gross_return": 0.0,
            "zero_return": 0.0,
            "kotak_return": 0.0,
            "ending_zero": starting_capital,
            "ending_kotak": starting_capital,
            "predicted_positive": 0,
            "direction_correct": 0,
        }

    raw_models, rel_models = v231.fit_raw_and_relative_models(labeled, features, replay_start)
    usable = seq.dropna(subset=features).copy()
    day = usable[(usable["timestamp"] >= replay_start) & (usable["timestamp"] < replay_end)].copy()
    counts = day.groupby("timestamp")["symbol"].nunique()
    times = list(counts[counts == len(DEFAULT_CONFIG.universe)].index)
    if not times:
        return {
            "date": date_str,
            "gate_active": True,
            "reason": "gate active but no complete replay bars",
            "cal_sessions": len(cal_dates),
            "trades": 0,
            "wins": 0,
            "gross_return": 0.0,
            "zero_return": 0.0,
            "kotak_return": 0.0,
            "ending_zero": starting_capital,
            "ending_kotak": starting_capital,
            "predicted_positive": 0,
            "direction_correct": 0,
        }

    cash_zero = starting_capital
    cash_kotak = starting_capital
    positions: dict[str, Position] = {}
    trade_log: list[dict] = []
    entries_by_session = {"OPEN": 0, "MID": 0, "LATE": 0}
    total_entries = 0

    for ts in times:
        exec_info = {s: next_open(bars[s], pd.Timestamp(ts)) for s in DEFAULT_CONFIG.universe}
        if any(v is None for v in exec_info.values()):
            continue
        exec_ts = min(v[0] for v in exec_info.values() if v is not None)
        prices = {s: v[1] for s, v in exec_info.items() if v is not None}

        # Exit positions whose chosen horizon has matured.
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
                "symbol": symbol,
                "entry_ts": p.entry_ts,
                "exit_ts": exec_ts,
                "horizon": p.horizon,
                "pred_net": p.predicted_net_kotak,
                "actual_gross": px / p.entry_price - 1.0,
                "net_zero": gross_pnl - p.buy_cost_zero - sell_zero,
                "net_kotak": gross_pnl - p.buy_cost_kotak - sell_kotak,
            })
            del positions[symbol]

        snap = day[day["timestamp"] == ts].copy().sort_values("symbol")
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
        for r in eligible.itertuples(index=False):
            symbol = str(r.symbol)
            if symbol in positions or len(positions) >= MAX_POSITIONS:
                continue
            if total_entries >= gates.max_daily_entries:
                break
            if session is None or gates.session_caps is None:
                break
            if entries_by_session[session] >= gates.session_caps.get(session, 0):
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
                symbol=symbol,
                qty=qty,
                entry_price=px,
                entry_ts=exec_ts,
                planned_exit_ts=planned_exit,
                horizon=h,
                predicted_gross=pred_gross,
                predicted_net_zero=nz,
                predicted_net_kotak=nk,
                buy_cost_zero=buy_zero,
                buy_cost_kotak=buy_kotak,
            )
            total_entries += 1
            entries_by_session[session] += 1

    # Mandatory intraday square-off at the final available open.
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
            "symbol": symbol,
            "entry_ts": p.entry_ts,
            "exit_ts": exit_ts,
            "horizon": p.horizon,
            "pred_net": p.predicted_net_kotak,
            "actual_gross": px / p.entry_price - 1.0,
            "net_zero": gross_pnl - p.buy_cost_zero - sell_zero,
            "net_kotak": gross_pnl - p.buy_cost_kotak - sell_kotak,
        })
        del positions[symbol]

    trades = pd.DataFrame(trade_log)
    gross_pnl = float((trades["actual_gross"] * 0).sum()) if trades.empty else float(
        trades["net_kotak"].sum()
    )
    # Reconstruct gross capital effect from actual price P&L by adding all charged costs back.
    # For summary accuracy we use actual-gross direction directly from each trade.
    zero_ret = cash_zero / starting_capital - 1.0
    kotak_ret = cash_kotak / starting_capital - 1.0

    if trades.empty:
        wins = 0
        direction_correct = 0
        predicted_positive = 0
        gross_return = 0.0
    else:
        wins = int((trades["net_kotak"] > 0).sum())
        predicted_positive = int((trades["pred_net"] > 0).sum())
        direction_correct = int(((trades["pred_net"] > 0) & (trades["actual_gross"] > 0)).sum())
        # Approximate gross portfolio return by adding actual statutory+broker cost drag back to Kotak ending capital.
        # More directly, gross P&L equals net P&L plus order costs; derive from zero-vs-kotak not enough,
        # so report mean trade gross separately and use zero/kotak returns for capital economics.
        gross_return = float(trades["actual_gross"].mean())

    return {
        "date": date_str,
        "gate_active": True,
        "reason": gates.reason,
        "cal_sessions": len(cal_dates),
        "trades": int(len(trades)),
        "wins": wins,
        "gross_return": gross_return,
        "zero_return": zero_ret,
        "kotak_return": kotak_ret,
        "ending_zero": cash_zero,
        "ending_kotak": cash_kotak,
        "predicted_positive": predicted_positive,
        "direction_correct": direction_correct,
        "gate_pred_net": gates.min_predicted_net,
        "gate_rel": gates.min_relative_score,
        "gate_agreement": gates.min_agreement,
        "gate_daily_cap": gates.max_daily_entries,
    }


def main() -> None:
    print("Share-Trading-AI v24.1 robust multi-day walk-forward abstention replay")
    print("NO BROKER ORDERS ARE SENT. Historical research only.")
    print("Same v24 principle each day: trade only if a positive-after-cost gate survives two earlier time blocks; otherwise CASH.")

    live = build_live_panel(load_events())
    seq, features = feature_block(live)
    labeled = add_horizon_labels(seq)
    bars = execution_series(seq)

    dates = available_dates(seq)
    if len(dates) < TEST_SESSIONS + 10:
        raise RuntimeError(f"Not enough complete sessions for walk-forward replay: {len(dates)}")
    test_dates = dates[-TEST_SESSIONS:]
    print(f"Replay window: {test_dates[0]} -> {test_dates[-1]} ({len(test_dates)} sessions)")

    capital_zero = STARTING_CAPITAL
    capital_kotak = STARTING_CAPITAL
    results = []

    for i, date_str in enumerate(test_dates, start=1):
        print(f"\n===== SESSION {i}/{len(test_dates)}: {date_str} =====")
        # Use the Kotak capital as common sizing capital; maintain both accounting profiles from same starting base.
        # To keep both profiles comparable, replay from the lower/common carried capital each day.
        common_start = min(capital_zero, capital_kotak)
        r = replay_one_day(date_str, common_start, seq, features, labeled, bars)
        capital_zero = r["ending_zero"]
        capital_kotak = r["ending_kotak"]
        results.append(r)

        if not r["gate_active"]:
            print(f"gate=OFF | NO TRADE | reason={r['reason']}")
        else:
            print(
                f"gate=ON trades={r['trades']:2d} wins={r['wins']:2d} "
                f"zero={r['zero_return']:+.3%} kotak={r['kotak_return']:+.3%} "
                f"gate_pred_net={r.get('gate_pred_net', np.nan):+.3%} "
                f"rel={r.get('gate_rel', np.nan):+.3f} agree={r.get('gate_agreement', 0)}/5 cap={r.get('gate_daily_cap', 0)}"
            )

    out = pd.DataFrame(results)
    active = out[out["gate_active"]]
    traded = out[out["trades"] > 0]
    total_trades = int(out["trades"].sum())
    total_wins = int(out["wins"].sum())
    pred_pos = int(out["predicted_positive"].sum())
    dir_correct = int(out["direction_correct"].sum())

    print("\nV24.1 LAST-MONTH ROBUST SUMMARY")
    print(f"  sessions tested: {len(out)}")
    print(f"  gate active sessions: {int(out['gate_active'].sum())} ({out['gate_active'].mean():.2%})")
    print(f"  full CASH / abstention sessions: {int((~out['gate_active']).sum())} ({(~out['gate_active']).mean():.2%})")
    print(f"  sessions with at least one actual trade: {len(traded)} ({len(traded)/len(out):.2%})")
    print(f"  total completed trades: {total_trades}")
    print(f"  average trades / tested session: {total_trades/len(out):.2f}")
    print(f"  profitable trades after Kotak costs: {total_wins}/{total_trades} ({total_wins/total_trades:.2%})" if total_trades else "  profitable trades after Kotak costs: N/A (no trades)")
    print(f"  positive-direction accuracy: {dir_correct}/{pred_pos} ({dir_correct/pred_pos:.2%})" if pred_pos else "  positive-direction accuracy: N/A")
    print(f"  profitable traded sessions after Kotak costs: {(traded['kotak_return'] > 0).mean():.2%}" if len(traded) else "  profitable traded sessions after Kotak costs: N/A")
    print(f"  average Kotak return / tested session: {out['kotak_return'].mean():+.3%}")
    print(f"  average zero-brokerage return / tested session: {out['zero_return'].mean():+.3%}")

    # Compound using actual per-day returns independently to avoid mixing the two profile capital bases.
    compounded_zero = float(np.prod(1.0 + out["zero_return"].to_numpy()) - 1.0)
    compounded_kotak = float(np.prod(1.0 + out["kotak_return"].to_numpy()) - 1.0)
    final_zero = STARTING_CAPITAL * (1.0 + compounded_zero)
    final_kotak = STARTING_CAPITAL * (1.0 + compounded_kotak)
    print(f"  ZERO-brokerage compounded capital: INR {final_zero:,.2f} | return={compounded_zero:+.3%}")
    print(f"  KOTAK-profile compounded capital: INR {final_kotak:,.2f} | return={compounded_kotak:+.3%}")

    print("\nDAILY RESULTS")
    for r in out.itertuples(index=False):
        state = "ON " if r.gate_active else "OFF"
        print(
            f"  {r.date} gate={state} trades={int(r.trades):2d} wins={int(r.wins):2d} "
            f"zero={r.zero_return:+.3%} kotak={r.kotak_return:+.3%}"
        )

    print("\nInterpretation:")
    print("  Accuracy is measured only on trades the robust gate actually allowed; abstention is not counted as a failed prediction.")
    print("  Profitability is the compounded after-cost capital result across all tested days, including zero-return CASH days.")
    print("  Do not loosen the gate merely to increase trade count after seeing this month; judge the frozen process on aggregate results.")


if __name__ == "__main__":
    main()
