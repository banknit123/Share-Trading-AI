from __future__ import annotations

import numpy as np
import pandas as pd

from run_intraday_portfolio_v21 import STARTING_CAPITAL, add_horizon_labels, execution_series
from run_forward_paper_observer_v19 import build_live_panel, feature_block
from run_historical_multimodal_v10 import load_events
import run_robust_multiday_v24_1 as v241


def replay_one_day_safe(date_str, starting_capital, seq, features, labeled, bars):
    """Run the frozen v24.1 day replay, treating its known empty-trade reporting bug as a zero-return day."""
    try:
        return v241.replay_one_day(date_str, starting_capital, seq, features, labeled, bars)
    except KeyError as exc:
        if str(exc).strip("'") != "actual_gross":
            raise
        return {
            "date": date_str,
            "gate_active": True,
            "reason": "robust gate active, but no completed trade passed execution constraints",
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
            "gate_pred_net": np.nan,
            "gate_rel": np.nan,
            "gate_agreement": 0,
            "gate_daily_cap": 0,
        }


def main() -> None:
    print("Share-Trading-AI v24.2 robust multi-day walk-forward abstention replay")
    print("NO BROKER ORDERS ARE SENT. Historical research only.")
    print("Same frozen v24.1 strategy; v24.2 only fixes active-gate / zero-trade day reporting.")

    live = build_live_panel(load_events())
    seq, features = feature_block(live)
    labeled = add_horizon_labels(seq)
    bars = execution_series(seq)

    dates = v241.available_dates(seq)
    if len(dates) < v241.TEST_SESSIONS + 10:
        raise RuntimeError(f"Not enough complete sessions for walk-forward replay: {len(dates)}")
    test_dates = dates[-v241.TEST_SESSIONS:]
    print(f"Replay window: {test_dates[0]} -> {test_dates[-1]} ({len(test_dates)} sessions)")

    capital_zero = STARTING_CAPITAL
    capital_kotak = STARTING_CAPITAL
    results = []

    for i, date_str in enumerate(test_dates, start=1):
        print(f"\n===== SESSION {i}/{len(test_dates)}: {date_str} =====")
        common_start = min(capital_zero, capital_kotak)
        r = replay_one_day_safe(date_str, common_start, seq, features, labeled, bars)
        capital_zero = r["ending_zero"]
        capital_kotak = r["ending_kotak"]
        results.append(r)

        if not r["gate_active"]:
            print(f"gate=OFF | NO TRADE | reason={r['reason']}")
        else:
            extra = ""
            if r["trades"] == 0:
                extra = f" | {r['reason']}"
            print(
                f"gate=ON trades={r['trades']:2d} wins={r['wins']:2d} "
                f"zero={r['zero_return']:+.3%} kotak={r['kotak_return']:+.3%} "
                f"gate_pred_net={r.get('gate_pred_net', np.nan):+.3%} "
                f"rel={r.get('gate_rel', np.nan):+.3f} agree={r.get('gate_agreement', 0)}/5 "
                f"cap={r.get('gate_daily_cap', 0)}{extra}"
            )

    out = pd.DataFrame(results)
    traded = out[out["trades"] > 0]
    total_trades = int(out["trades"].sum())
    total_wins = int(out["wins"].sum())
    pred_pos = int(out["predicted_positive"].sum())
    dir_correct = int(out["direction_correct"].sum())

    print("\nV24.2 LAST-MONTH ROBUST SUMMARY")
    print(f"  sessions tested: {len(out)}")
    print(f"  gate active sessions: {int(out['gate_active'].sum())} ({out['gate_active'].mean():.2%})")
    print(f"  full CASH / gate-OFF sessions: {int((~out['gate_active']).sum())} ({(~out['gate_active']).mean():.2%})")
    print(f"  active-gate but zero-trade sessions: {int(((out['gate_active']) & (out['trades'] == 0)).sum())}")
    print(f"  sessions with at least one actual trade: {len(traded)} ({len(traded)/len(out):.2%})")
    print(f"  total completed trades: {total_trades}")
    print(f"  average trades / tested session: {total_trades/len(out):.2f}")
    if total_trades:
        print(f"  profitable trades after Kotak costs: {total_wins}/{total_trades} ({total_wins/total_trades:.2%})")
        print(f"  positive-direction accuracy: {dir_correct}/{pred_pos} ({dir_correct/pred_pos:.2%})" if pred_pos else "  positive-direction accuracy: N/A")
    else:
        print("  profitable trades after Kotak costs: N/A (no trades)")
        print("  positive-direction accuracy: N/A")
    print(f"  profitable traded sessions after Kotak costs: {(traded['kotak_return'] > 0).mean():.2%}" if len(traded) else "  profitable traded sessions after Kotak costs: N/A")
    print(f"  average Kotak return / tested session: {out['kotak_return'].mean():+.3%}")
    print(f"  average zero-brokerage return / tested session: {out['zero_return'].mean():+.3%}")

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
    print("  Accuracy is measured only on completed trades. Gate-OFF and active-gate/zero-trade days preserve capital.")
    print("  Profitability is compounded across all tested sessions, including zero-return abstention days.")


if __name__ == "__main__":
    main()
