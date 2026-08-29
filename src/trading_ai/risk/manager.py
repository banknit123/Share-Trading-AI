from __future__ import annotations

from dataclasses import dataclass

from trading_ai.config import TradingConfig


@dataclass
class RiskState:
    starting_equity: float
    current_equity: float
    realized_pnl_today: float = 0.0


class RiskManager:
    def __init__(self, config: TradingConfig):
        self.config = config

    def can_trade(self, state: RiskState) -> tuple[bool, str]:
        max_loss = state.starting_equity * self.config.max_daily_loss_pct
        if state.realized_pnl_today <= -max_loss:
            return False, "Daily loss limit reached"
        return True, "OK"

    def max_position_value(self, equity: float) -> float:
        return equity * self.config.max_position_pct

    def position_size(self, equity: float, price: float) -> int:
        if price <= 0:
            return 0
        return int(self.max_position_value(equity) // price)
