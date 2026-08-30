from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.market_data import download_history
from trading_ai.features.intraday_v2 import HORIZON_BARS, add_intraday_features_v2
from trading_ai.models.intraday_v2 import IntradayModelV2


@dataclass
class EvalResult:
    model: str
    threshold: float
    val_trades: int
    val_return: float
    test_trades: int
    test_return: float
    test_win_rate: float


def simulate(scored: pd.DataFrame, target: str, bars: int, threshold: float, cost: float = 0.0012):
    equity = 1.0
    trades = []
    i = 0
    while i < len(scored) - bars:
        row = scored.iloc[i]
        if float(row["probability_up"]) >= threshold and float(row["expected_return"]) > cost:
            net = float(row[target]) - cost
            trades.append(net)
            equity *= 1 + net
            i += bars
        else:
            i += 1
    return len(trades), float(equity - 1), float(np.mean([r > 0 for r in trades])) if trades else 0.0


def evaluate_symbol(symbol: str, horizon: str) -> EvalResult:
    raw = download_history(symbol, period="60d", interval="5m")
    df = add_intraday_features_v2(raw)
    target = f"future_return_{horizon}"
    bars = HORIZON_BARS[horizon]

    n = len(df)
    i1 = int(n * 0.60)
    i2 = int(n * 0.80)
    train, val, test = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]

    candidates = []
    for kind in ("logistic", "hgb"):
        model = IntradayModelV2(target, kind=kind).fit(train)
        val_scored = model.predict(val)
        for threshold in (0.52, 0.55, 0.58, 0.60, 0.62, 0.65):
            trades, ret, win = simulate(val_scored, target, bars, threshold)
            if trades >= 5:
                score = ret - max(0.0, 0.45 - win) * 0.02
                candidates.append((score, ret, trades, win, kind, threshold))

    if not candidates:
        return EvalResult("none", 0.0, 0, 0.0, 0, 0.0, 0.0)

    candidates.sort(reverse=True)
    _, val_ret, val_trades, _, kind, threshold = candidates[0]

    train_plus_val = df.iloc[:i2]
    final_model = IntradayModelV2(target, kind=kind).fit(train_plus_val)
    test_scored = final_model.predict(test)
    test_trades, test_ret, test_win = simulate(test_scored, target, bars, threshold)
    return EvalResult(kind, threshold, val_trades, val_ret, test_trades, test_ret, test_win)


def main() -> None:
    print("Share-Trading-AI intraday v2 robust test")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)
    print("Models: logistic + hist-gradient-boosting")
    print("Selection: validation only; test remains untouched")

    summary = {h: [] for h in HORIZON_BARS}
    for symbol in DEFAULT_CONFIG.universe:
        print(f"\n{symbol}")
        for horizon in HORIZON_BARS:
            try:
                r = evaluate_symbol(symbol, horizon)
                summary[horizon].append(r)
                print(
                    f"  {horizon:3} model={r.model:8} threshold={r.threshold:.2f} "
                    f"val_trades={r.val_trades:3d} val_ret={r.val_return:7.2%} "
                    f"test_trades={r.test_trades:3d} test_ret={r.test_return:7.2%} "
                    f"test_win={r.test_win_rate:6.2%}"
                )
            except Exception as exc:
                print(f"  {horizon:3} ERROR: {exc}")

    print("\nCross-symbol v2 untouched-test summary")
    for horizon, rows in summary.items():
        profitable = sum(r.test_return > 0 for r in rows)
        active = [r for r in rows if r.test_trades > 0]
        avg_ret = float(np.mean([r.test_return for r in rows])) if rows else 0.0
        med_ret = float(np.median([r.test_return for r in rows])) if rows else 0.0
        total_trades = sum(r.test_trades for r in rows)
        med_win = float(np.median([r.test_win_rate for r in active])) if active else 0.0
        print(
            f"{horizon}: profitable={profitable}/{len(rows)} avg_return={avg_ret:.2%} "
            f"median_return={med_ret:.2%} total_trades={total_trades} median_win={med_win:.2%}"
        )

    print("\nResearch-only. Live trading must remain disabled until repeated out-of-sample tests pass.")


if __name__ == "__main__":
    main()
