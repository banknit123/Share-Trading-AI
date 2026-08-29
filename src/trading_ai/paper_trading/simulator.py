from __future__ import annotations

from dataclasses import dataclass, field

from trading_ai.config import TradingConfig
from trading_ai.risk.manager import RiskManager, RiskState


@dataclass
class PaperPosition:
    symbol: str
    quantity: int
    entry_price: float


@dataclass
class PaperAccount:
    cash: float
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    realized_pnl_today: float = 0.0


class PaperBroker:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.account = PaperAccount(cash=config.initial_capital)
        self.risk = RiskManager(config)

    def equity(self, prices: dict[str, float]) -> float:
        value = self.account.cash
        for symbol, position in self.account.positions.items():
            value += position.quantity * prices.get(symbol, position.entry_price)
        return value

    def buy(self, symbol: str, price: float, prices: dict[str, float]) -> dict:
        eq = self.equity(prices)
        state = RiskState(self.config.initial_capital, eq, self.account.realized_pnl_today)
        allowed, reason = self.risk.can_trade(state)
        if not allowed:
            return {"status": "blocked", "reason": reason}

        qty = self.risk.position_size(eq, price)
        if qty <= 0:
            return {"status": "blocked", "reason": "Position size is zero"}
        required = qty * price
        if required > self.account.cash:
            qty = int(self.account.cash // price)
        if qty <= 0:
            return {"status": "blocked", "reason": "Insufficient paper cash"}

        self.account.cash -= qty * price
        self.account.positions[symbol] = PaperPosition(symbol, qty, price)
        return {"status": "filled", "side": "BUY", "symbol": symbol, "quantity": qty, "price": price}

    def sell(self, symbol: str, price: float) -> dict:
        position = self.account.positions.get(symbol)
        if not position:
            return {"status": "blocked", "reason": "No long position to sell"}
        proceeds = position.quantity * price
        pnl = position.quantity * (price - position.entry_price)
        self.account.cash += proceeds
        self.account.realized_pnl_today += pnl
        del self.account.positions[symbol]
        return {"status": "filled", "side": "SELL", "symbol": symbol, "quantity": position.quantity, "price": price, "realized_pnl": pnl}
