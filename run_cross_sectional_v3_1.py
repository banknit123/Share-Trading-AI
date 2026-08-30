from __future__ import annotations

import numpy as np

from trading_ai.config import DEFAULT_CONFIG
from run_cross_sectional_v3 import build_panel, split_by_time, fit_model, score

BARS = 6
ROUND_TRIP_COST = 0.0012
THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.40, 0.56, 0.02))
MIN_VALIDATION_TRADES = 5


def simulate(scored, threshold: float):
    equity = 1.0
    returns = []
    details = []
    all_times = sorted(scored["timestamp"].unique())
    i = 0
    while i < len(all_times) - BARS:
        ts = all_times[i]
        snap = scored[scored["timestamp"] == ts]
        if snap.empty or float(snap["benchmark_ret_6"].iloc[0]) <= 0:
            i += 1
            continue

        candidate = snap.sort_values(["probability", "rs_6"], ascending=False).iloc[0]
        p = float(candidate["probability"])
        rs6 = float(candidate["rs_6"])
        if rs6 <= 0 or p < threshold:
            i += 1
            continue

        gross = float(candidate["future_return_30m"])
        net = gross - ROUND_TRIP_COST
        returns.append(net)
        details.append((ts, str(candidate["symbol"]), p, rs6, net))
        equity *= 1 + net
        i += BARS

    wins = sum(r > 0 for r in returns)
    return {
        "trades": len(returns),
        "return": equity - 1,
        "win_rate": wins / len(returns) if returns else 0.0,
        "avg_trade": float(np.mean(returns)) if returns else 0.0,
        "details": details,
    }


def choose_threshold(val_scored):
    candidates = []
    for threshold in THRESHOLDS:
        r = simulate(val_scored, threshold)
        print(
            f"  val threshold={threshold:.2f} trades={r['trades']:3d} "
            f"return={r['return']:7.2%} win={r['win_rate']:7.2%} avg={r['avg_trade']:7.3%}"
        )
        if r["trades"] >= MIN_VALIDATION_TRADES and r["avg_trade"] > 0:
            score_value = r["return"] - 0.25 * abs(min(r["avg_trade"], 0))
            candidates.append((score_value, threshold, r))

    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, threshold, result = candidates[0]
    return threshold, result


def main() -> None:
    print("Share-Trading-AI v3.1 evidence-based confidence test")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)
    print("Keeps market-regime and relative-strength filters unchanged.")
    print("Only the confidence layer is tested at 0.40-0.54, based on v3 diagnostics.")

    panel = build_panel()
    train, val, test = split_by_time(panel)
    model = fit_model(train)
    val_scored = score(model, val)
    test_scored = score(model, test)

    print(f"Rows: train={len(train)} validation={len(val)} test={len(test)}")
    print("\nValidation threshold sweep:")
    threshold, val_result = choose_threshold(val_scored)

    if threshold is None:
        print("\nNo validation configuration met the minimum-trades + positive-average-trade rule.")
        print("Research gate: FAIL - RESEARCH ONLY")
        return

    test_result = simulate(test_scored, threshold)
    print(
        f"\nSelected threshold={threshold:.2f} | validation trades={val_result['trades']} "
        f"return={val_result['return']:.2%} win={val_result['win_rate']:.2%} avg_trade={val_result['avg_trade']:.3%}"
    )
    print(
        f"UNTOUCHED TEST | trades={test_result['trades']} return={test_result['return']:.2%} "
        f"win={test_result['win_rate']:.2%} avg_trade={test_result['avg_trade']:.3%}"
    )

    ready = (
        test_result["trades"] >= 12
        and test_result["return"] > 0
        and test_result["win_rate"] >= 0.50
        and test_result["avg_trade"] > 0
    )
    print("\nResearch gate:", "PASS FOR PAPER-TRADING CANDIDATE" if ready else "FAIL - RESEARCH ONLY")
    print("Live execution remains disabled regardless of this run.")


if __name__ == "__main__":
    main()
