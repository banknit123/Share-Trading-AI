from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_ai.features.intraday import HORIZON_BARS
from trading_ai.models.intraday import IntradayDirectionModel


@dataclass
class SliceResult:
    trades: int
    strategy_return: float
    win_rate: float
    max_drawdown: float


@dataclass
class OptimizedIntradayResult:
    horizon: str
    threshold: float
    validation: SliceResult
    test: SliceResult
    test_buy_hold_return: float


def _max_drawdown(curve: list[float]) -> float:
    if not curve:
        return 0.0
    s = pd.Series(curve, dtype=float)
    peak = s.cummax()
    dd = s / peak - 1
    return float(-dd.min())


def _simulate(scored: pd.DataFrame, target: str, bars: int, threshold: float, round_trip_cost: float) -> SliceResult:
    equity = 1.0
    curve: list[float] = []
    returns: list[float] = []
    i = 0
    while i < len(scored) - bars:
        row = scored.iloc[i]
        expected_after_cost = float(row["expected_return"]) - round_trip_cost
        if float(row["probability_up"]) >= threshold and expected_after_cost > 0:
            net = float(row[target]) - round_trip_cost
            returns.append(net)
            equity *= 1 + net
            curve.append(equity)
            i += bars
        else:
            curve.append(equity)
            i += 1
    wins = sum(r > 0 for r in returns)
    return SliceResult(
        trades=len(returns),
        strategy_return=float(equity - 1),
        win_rate=float(wins / len(returns)) if returns else 0.0,
        max_drawdown=_max_drawdown(curve),
    )


def optimize_intraday_threshold(
    df: pd.DataFrame,
    horizon: str,
    thresholds: tuple[float, ...] = (0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65),
    round_trip_cost: float = 0.0012,
    min_validation_trades: int = 5,
) -> OptimizedIntradayResult:
    if horizon not in HORIZON_BARS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    if len(df) < 1000:
        raise ValueError("Insufficient intraday history for train/validation/test optimisation")

    target = f"future_return_{horizon}"
    bars = HORIZON_BARS[horizon]
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    selector_model = IntradayDirectionModel(target).fit(train)
    scored_validation = selector_model.predict(validation)

    candidates: list[tuple[float, SliceResult]] = []
    for threshold in thresholds:
        result = _simulate(scored_validation, target, bars, threshold, round_trip_cost)
        candidates.append((threshold, result))

    eligible = [(t, r) for t, r in candidates if r.trades >= min_validation_trades]
    pool = eligible if eligible else candidates
    best_threshold, best_validation = max(
        pool,
        key=lambda item: (item[1].strategy_return, item[1].win_rate, -item[1].max_drawdown),
    )

    final_train = df.iloc[:val_end].copy()
    final_model = IntradayDirectionModel(target).fit(final_train)
    scored_test = final_model.predict(test)
    test_result = _simulate(scored_test, target, bars, best_threshold, round_trip_cost)

    test_buy_hold = 0.0
    if len(test) > bars:
        test_buy_hold = float(test["Close"].iloc[-1] / test["Close"].iloc[0] - 1)

    return OptimizedIntradayResult(
        horizon=horizon,
        threshold=best_threshold,
        validation=best_validation,
        test=test_result,
        test_buy_hold_return=test_buy_hold,
    )
