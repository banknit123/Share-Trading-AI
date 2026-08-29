from __future__ import annotations

from trading_ai.config import TradingConfig


class KotakNeoBroker:
    """Safety-locked placeholder for future Kotak Neo integration.

    No live API calls are implemented here yet. Live execution must remain
    disabled until credentials, broker API terms, order validation, kill
    switches, and paper-trading acceptance criteria are completed.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def place_order(self, *args, **kwargs):
        if not self.config.live_trading_enabled:
            raise RuntimeError("Live trading is disabled by configuration")
        raise NotImplementedError("Kotak Neo live execution is not implemented yet")
