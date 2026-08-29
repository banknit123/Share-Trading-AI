# Share-Trading-AI

Research, backtesting and paper-trading platform for Indian equities.

## Current status

Implemented:
- Historical OHLCV ingestion for NSE symbols via `yfinance`
- Technical feature engineering
- Baseline direction model
- Confidence/expected-return signal thresholds
- Cost/slippage-aware walk-forward backtesting
- Risk limits and position sizing
- Paper-trading simulator
- Safety-locked Kotak Neo adapter placeholder

Live trading is **disabled by default** and the current market-data source is for research/prototyping, not exchange-grade live execution.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python run_backtest.py
```

The first research universe is a small set of highly liquid NSE names plus NIFTY 50 as benchmark context.

## Next milestones

1. Validate the backtest across multiple time windows.
2. Add 5-minute/15-minute intraday data and 10/30/60-minute prediction targets.
3. Add benchmark/regime and news-sentiment features.
4. Add persistent paper-trade logging and a daily report.
5. Define acceptance gates before any broker execution is enabled.
