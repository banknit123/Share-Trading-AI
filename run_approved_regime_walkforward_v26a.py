from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_forward_paper_observer_v19 import build_live_panel, feature_block
from run_historical_multimodal_v10 import load_events
from run_intraday_portfolio_v21 import STARTING_CAPITAL
from run_stock_horizon_regime_v25 import build_observations

# Frozen discovery from v25. This script validates it; it does NOT search for a better rule.
SYMBOL = "TCS.NS"
HORIZON = "120m"
MARKET_REGIME = "MARKET_DOWN"
TREND_STATE = "UPTREND"

TEST_SESSIONS = 20
TRAILING_SESSIONS = 20
MIN_HISTORY_OBS = 20
MIN_HISTORY_DAYS = 6
MIN_HISTORY_WIN_RATE = 0.50
MIN_HISTORY_POSITIVE_DAY_RATE = 0.55
CAPITAL_FRACTION = 0.30
SLIPPAGE_BPS_EACH_SIDE = [0, 1, 2, 5]
OUT_DIR = Path("data/v26a")


@dataclass
class GateStats:
    active: bool
    n: int
    days: int
    mean_net: float
    median_net: float
    win_rate: float
    positive_day_rate: float
    reason: str


def trailing_gate(history: pd.DataFrame) -> GateStats:
    if history.empty:
        return GateStats(False, 0, 0, np.nan, np.nan, np.nan, np.nan, "no prior matching observations")
    by_day = history.groupby("date")["net_kotak"].mean()
    n = len(history)
    days = history["date"].nunique()
    mean_net = float(history["net_kotak"].mean())
    median_net = float(history["net_kotak"].median())
    win_rate = float((history["net_kotak"] > 0).mean())
    positive_day_rate = float((by_day > 0).mean())

    checks = [
        (n >= MIN_HISTORY_OBS, f"obs {n}<{MIN_HISTORY_OBS}"),
        (days >= MIN_HISTORY_DAYS, f"days {days}<{MIN_HISTORY_DAYS}"),
        (mean_net > 0, f"mean_net {mean_net:+.3%}<=0"),
        (median_net > 0, f"median_net {median_net:+.3%}<=0"),
        (win_rate > MIN_HISTORY_WIN_RATE, f"win {win_rate:.1%}<={MIN_HISTORY_WIN_RATE:.0%}"),
        (positive_day_rate >= MIN_HISTORY_POSITIVE_DAY_RATE, f"positive_days {positive_day_rate:.1%}<{MIN_HISTORY_POSITIVE_DAY_RATE:.0%}"),
    ]
    failed = [reason for ok, reason in checks if not ok]
    return GateStats(
        not failed, n, days, mean_net, median_net, win_rate, positive_day_rate,
        "passed" if not failed else "; ".join(failed),
    )


def add_slippage(net_return: float, bps_each_side: int) -> float:
    # Approximate two-sided slippage drag on return. Costs already exist inside net_kotak.
    return float(net_return - 2.0 * bps_each_side / 10000.0)


def non_overlapping_day_trades(day: pd.DataFrame) -> pd.DataFrame:
    """Select chronological TCS opportunities while allowing at most one open position.

    v25 observations already use executable next-open entry and future-open exit.
    """
    if day.empty:
        return day.copy()
    d = day.sort_values(["entry_ts", "exit_ts"]).copy()
    chosen = []
    last_exit = None
    for r in d.itertuples(index=False):
        entry_ts = pd.Timestamp(r.entry_ts)
        exit_ts = pd.Timestamp(r.exit_ts)
        if last_exit is not None and entry_ts < last_exit:
            continue
        chosen.append(r._asdict())
        last_exit = exit_ts
    return pd.DataFrame(chosen)


def main() -> None:
    print("Share-Trading-AI v26A Approved-Regime Walk-Forward Validator")
    print("NO BROKER ORDERS ARE SENT. Historical research only.")
    print("Frozen v25 regime: TCS.NS | 120m | MARKET_DOWN | UPTREND")
    print("The rule is NOT re-optimized here. Each test day is enabled only from earlier matching history.")
    print("Important: because v25 discovered this regime using the broader 59-session sample, this is a temporal robustness test, not fully independent discovery OOS proof.")

    live = build_live_panel(load_events())
    seq, _ = feature_block(live)
    obs = build_observations(seq)
    if obs.empty:
        raise RuntimeError("No v25-style observations available")

    rule = obs[
        (obs["symbol"] == SYMBOL)
        & (obs["horizon"] == HORIZON)
        & (obs["market_regime"] == MARKET_REGIME)
        & (obs["trend_state"] == TREND_STATE)
    ].copy()
    if rule.empty:
        raise RuntimeError("Frozen TCS regime has no observations in current data")

    all_dates = sorted(obs["date"].unique())
    if len(all_dates) <= TEST_SESSIONS:
        raise RuntimeError(f"Need more than {TEST_SESSIONS} sessions; found {len(all_dates)}")
    test_dates = all_dates[-TEST_SESSIONS:]

    print(f"\nAll data sessions: {all_dates[0]} -> {all_dates[-1]} ({len(all_dates)})")
    print(f"Frozen-rule observations: {len(rule)} across {rule['date'].nunique()} sessions")
    print(f"Walk-forward test window: {test_dates[0]} -> {test_dates[-1]} ({len(test_dates)} sessions)")

    daily_rows = []
    trade_rows = []
    capitals = {bps: float(STARTING_CAPITAL) for bps in SLIPPAGE_BPS_EACH_SIDE}
    peaks = {bps: float(STARTING_CAPITAL) for bps in SLIPPAGE_BPS_EACH_SIDE}
    max_dd = {bps: 0.0 for bps in SLIPPAGE_BPS_EACH_SIDE}

    for i, date_str in enumerate(test_dates, start=1):
        date_idx = all_dates.index(date_str)
        prior_dates = all_dates[max(0, date_idx - TRAILING_SESSIONS):date_idx]
        history = rule[rule["date"].isin(prior_dates)].copy()
        gate = trailing_gate(history)

        today_all = rule[rule["date"] == date_str].copy()
        today = non_overlapping_day_trades(today_all) if gate.active else today_all.iloc[0:0].copy()
        trades_today = len(today)
        wins_today = int((today["net_kotak"] > 0).sum()) if trades_today else 0
        mean_today = float(today["net_kotak"].mean()) if trades_today else 0.0

        day_rets = {}
        for bps in SLIPPAGE_BPS_EACH_SIDE:
            start_cap = capitals[bps]
            for tr in today.itertuples(index=False):
                trade_net = add_slippage(float(tr.net_kotak), bps)
                allocated = capitals[bps] * CAPITAL_FRACTION
                capitals[bps] += allocated * trade_net
                trade_rows.append({
                    "date": date_str,
                    "symbol": tr.symbol,
                    "horizon": tr.horizon,
                    "entry_ts": tr.entry_ts,
                    "exit_ts": tr.exit_ts,
                    "gross_return": tr.gross_return,
                    "net_kotak": tr.net_kotak,
                    "slippage_bps_each_side": bps,
                    "net_after_slippage": trade_net,
                })
            day_rets[bps] = capitals[bps] / start_cap - 1.0 if start_cap else 0.0
            peaks[bps] = max(peaks[bps], capitals[bps])
            dd = capitals[bps] / peaks[bps] - 1.0
            max_dd[bps] = min(max_dd[bps], dd)

        daily_rows.append({
            "date": date_str,
            "gate_active": gate.active,
            "gate_reason": gate.reason,
            "history_n": gate.n,
            "history_days": gate.days,
            "history_mean_net": gate.mean_net,
            "history_win_rate": gate.win_rate,
            "history_positive_day_rate": gate.positive_day_rate,
            "raw_opportunities": len(today_all),
            "trades": trades_today,
            "wins": wins_today,
            "mean_trade_net": mean_today,
            **{f"day_return_{bps}bps": day_rets[bps] for bps in SLIPPAGE_BPS_EACH_SIDE},
        })

        print(
            f"{i:2d}/{len(test_dates)} {date_str} gate={'ON ' if gate.active else 'OFF'} "
            f"hist_n={gate.n:3d} hist_mean={gate.mean_net:+.3%} hist_win={gate.win_rate:.1%} "
            f"opps={len(today_all):2d} trades={trades_today:2d} wins={wins_today:2d} "
            f"net_mean={mean_today:+.3%}"
        )
        if not gate.active:
            print(f"     abstain reason: {gate.reason}")

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    base_trades = trades[trades["slippage_bps_each_side"] == 0].copy() if not trades.empty else pd.DataFrame()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_DIR / "v26a_daily.csv", index=False)
    trades.to_csv(OUT_DIR / "v26a_trades_slippage.csv", index=False)

    total_trades = int(daily["trades"].sum())
    total_wins = int(daily["wins"].sum())
    traded_days = daily[daily["trades"] > 0]

    print("\nV26A WALK-FORWARD SUMMARY")
    print(f"  test sessions: {len(daily)}")
    print(f"  gate-active sessions: {int(daily['gate_active'].sum())} ({daily['gate_active'].mean():.1%})")
    print(f"  abstention sessions: {int((~daily['gate_active']).sum())} ({(~daily['gate_active']).mean():.1%})")
    print(f"  raw matching opportunities on test days: {int(daily['raw_opportunities'].sum())}")
    print(f"  non-overlapping trades executed by validator: {total_trades}")
    if total_trades:
        print(f"  profitable trades after Kotak costs: {total_wins}/{total_trades} ({total_wins/total_trades:.1%})")
        print(f"  mean trade net after Kotak costs: {base_trades['net_kotak'].mean():+.3%}")
        print(f"  median trade net after Kotak costs: {base_trades['net_kotak'].median():+.3%}")
        print(f"  profitable traded sessions: {(traded_days['day_return_0bps'] > 0).mean():.1%}")
    else:
        print("  trade accuracy: N/A (no trade survived the rolling historical gate)")

    print("\n₹10 LAKH PORTFOLIO SENSITIVITY (30% capital per non-overlapping trade)")
    for bps in SLIPPAGE_BPS_EACH_SIDE:
        final_cap = capitals[bps]
        ret = final_cap / STARTING_CAPITAL - 1.0
        print(
            f"  slippage {bps} bps/side: final INR {final_cap:,.2f} | "
            f"return={ret:+.3%} | max_drawdown={max_dd[bps]:+.3%}"
        )

    print("\nDECISION GUIDE")
    if total_trades == 0:
        print("  FAIL / INCONCLUSIVE: the previously discovered regime did not remain active often enough in the forward-style window.")
    else:
        win = total_wins / total_trades
        mean_net = float(base_trades["net_kotak"].mean())
        robust_5bps = capitals[5] > STARTING_CAPITAL
        if win >= 0.60 and mean_net > 0 and robust_5bps:
            print("  PROMISING: >=60% trade win rate, positive mean after costs, and positive portfolio result even at 5 bps/side slippage.")
            print("  Next step: v26B specialist paper-trading engine, still NO live orders.")
        elif mean_net > 0 and capitals[2] > STARTING_CAPITAL:
            print("  WATCHLIST: positive edge, but robustness is not yet strong enough for specialist deployment.")
        else:
            print("  REJECT: the v25 TCS regime did not retain sufficient forward-style profitability/robustness.")

    print("\nFiles written:")
    print(f"  {OUT_DIR / 'v26a_daily.csv'}")
    print(f"  {OUT_DIR / 'v26a_trades_slippage.csv'}")


if __name__ == "__main__":
    main()
