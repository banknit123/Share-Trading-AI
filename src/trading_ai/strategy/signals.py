from __future__ import annotations

from dataclasses import dataclass

from trading_ai.config import TradingConfig
from trading_ai.models.baseline import Prediction


@dataclass(frozen=True)
class Signal:
    action: str
    confidence: float
    expected_return: float
    reason: str


def make_signal(prediction: Prediction, config: TradingConfig) -> Signal:
    p_up = prediction.probability_up
    exp_ret = prediction.expected_return

    if p_up >= config.minimum_confidence and exp_ret >= config.minimum_expected_return:
        return Signal("BUY", p_up, exp_ret, "Positive expected return with sufficient confidence")

    p_down = 1.0 - p_up
    if p_down >= config.minimum_confidence and exp_ret <= -config.minimum_expected_return:
        return Signal("SELL", p_down, exp_ret, "Negative expected return with sufficient confidence")

    return Signal("HOLD", max(p_up, p_down), exp_ret, "Signal threshold not met")
