from dataclasses import dataclass


@dataclass(frozen=True)
class TradingConfig:
    universe: tuple[str, ...] = (
        "RELIANCE.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "INFY.NS",
        "TCS.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "ITC.NS",
        "LT.NS",
        "AXISBANK.NS",
    )
    benchmark: str = "^NSEI"
    initial_capital: float = 100000.0
    max_position_pct: float = 0.10
    max_daily_loss_pct: float = 0.02
    minimum_confidence: float = 0.65
    minimum_expected_return: float = 0.003
    assumed_cost_bps_per_side: float = 7.5
    assumed_slippage_bps_per_side: float = 5.0
    live_trading_enabled: bool = False


DEFAULT_CONFIG = TradingConfig()
