from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_ai.features.intraday import HORIZON_BARS
from trading_ai.models.intraday import IntradayDirectionModel


@dataclass
class IntradayBacktestResult:
    horizon: str
    trades: int
    strategy_return: float
    buy_hold_return: float
    win_rate: float
    max_drawdown: float


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(-dd.min()) if len(dd) else 0.0


def intraday_backtest(
    df: pd.DataFrame,
    horizon: str,
    train_fraction: float = 0.70,
    probability_threshold: float = 0.58,
    round_trip_cost: float = 0.0012,
) -> IntradayBacktestResult:
    if horizon not in HORIZON_BARS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    if len(df) < 500:
        raise ValueError("Insufficient intraday history")

    target = f"future_return_{horizon}"
    split = int(len(df) * train_fraction)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    model = IntradayDirectionModel(target).fit(train)
    scored = model.predict(test)

    bars = HORIZON_BARS[horizon]
    equity = 1.0
    curve = []
    trade_returns: list[float] = []
    i = 0
    while i < len(scored) - bars:
        row = scored.iloc[i]
        expected_after_cost = float(row["expected_return"]) - round_trip_cost
        if float(row["probability_up"]) >= probability_threshold and expected_after_cost > 0:
            gross = float(row[target])
            net = gross - round_trip_cost
            trade_returns.append(net)
            equity *= 1 + net
            curve.append(equity)
            i += bars
        else:
            curve.append(equity)
            i += 1

    if len(test) > bars:
        buy_hold = float(test["Close"].iloc[-1] / test["Close"].iloc[0] - 1)
    else:
        buy_hold = 0.0
    wins = sum(r > 0 for r in trade_returns)
    win_rate = wins / len(trade_returns) if trade_returns else 0.0
    eq_series = pd.Series(curve if curve else [1.0], dtype=float)

    return IntradayBacktestResult(
        horizon=horizon,
        trades=len(trade_returns),
        strategy_return=float(equity - 1),
        buy_hold_return=buy_hold,
        win_rate=float(win_rate),
        max_drawdown=_max_drawdown(eq_series),
    )
