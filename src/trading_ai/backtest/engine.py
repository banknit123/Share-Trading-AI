from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_ai.config import TradingConfig
from trading_ai.models.baseline import BaselineDirectionModel
from trading_ai.strategy.signals import make_signal


@dataclass
class BacktestResult:
    starting_capital: float
    ending_capital: float
    total_return: float
    buy_hold_return: float
    trades: int
    win_rate: float
    max_drawdown: float


def _cost_fraction(config: TradingConfig) -> float:
    bps = config.assumed_cost_bps_per_side + config.assumed_slippage_bps_per_side
    return bps / 10000.0


def walk_forward_backtest(df: pd.DataFrame, config: TradingConfig, min_train: int = 120) -> BacktestResult:
    if len(df) <= min_train + 2:
        raise ValueError("Not enough data for walk-forward backtest")

    capital = config.initial_capital
    peak = capital
    max_dd = 0.0
    trade_returns: list[float] = []
    cost = _cost_fraction(config)

    for i in range(min_train, len(df) - 1):
        train = df.iloc[:i].copy()
        row = df.iloc[i]
        model = BaselineDirectionModel().fit(train)
        prediction = model.predict_one(row)
        signal = make_signal(prediction, config)
        next_ret = float(df.iloc[i]["future_return_1"])

        if signal.action == "BUY":
            net_ret = next_ret - 2 * cost
        elif signal.action == "SELL":
            net_ret = -next_ret - 2 * cost
        else:
            continue

        position_value = capital * config.max_position_pct
        pnl = position_value * net_ret
        capital += pnl
        trade_returns.append(net_ret)
        peak = max(peak, capital)
        dd = (capital - peak) / peak
        max_dd = min(max_dd, dd)

    start_close = float(df.iloc[min_train]["Close"])
    end_close = float(df.iloc[-1]["Close"])
    buy_hold = end_close / start_close - 1
    wins = sum(r > 0 for r in trade_returns)
    win_rate = wins / len(trade_returns) if trade_returns else 0.0

    return BacktestResult(
        starting_capital=config.initial_capital,
        ending_capital=capital,
        total_return=capital / config.initial_capital - 1,
        buy_hold_return=buy_hold,
        trades=len(trade_returns),
        win_rate=win_rate,
        max_drawdown=abs(max_dd),
    )
