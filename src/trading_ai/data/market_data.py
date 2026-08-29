from __future__ import annotations

import pandas as pd
import yfinance as yf


class MarketDataError(RuntimeError):
    pass


def download_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data for research/backtesting.

    Uses Yahoo Finance via yfinance. This is convenient for prototyping but is
    not an exchange-grade feed and must not be treated as authoritative for
    live execution.
    """
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise MarketDataError(f"No market data returned for {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    wanted = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in data.columns]
    data = data[wanted].copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data.dropna(subset=["Close"])


def download_universe(symbols: list[str] | tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            out[symbol] = download_history(symbol, period=period)
        except MarketDataError:
            continue
    return out
