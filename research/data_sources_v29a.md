# v29A Data Capability & Signal Source Plan

Purpose: improve the research information set before building another predictive model.

## Why v29A exists

Versions v22-v28 did not demonstrate a robust, transferable after-cost intraday edge from the available 5-minute OHLCV-derived features. v29A therefore changes the input data architecture rather than loosening thresholds or adding more indicator combinations.

## Priority historical data sources

### DhanHQ
- Official intraday historical API supports 1, 5, 15, 25 and 60-minute candles.
- Official documentation states intraday data is available for the last 5 years for active instruments.
- Response includes OHLC, volume and optional OI for supported derivatives.
- This is currently the best fit for the target 1-3 year NSE intraday research window if the account/API access is available.

### Zerodha Kite Connect
- Historical candle API supports minute, 3, 5, 10, 15, 30 and 60-minute candles.
- Documentation describes archived candle data spanning back several years.
- Futures historical requests can include OI. Continuous futures support is limited in scope and expired instrument-token handling requires care.
- Strong candidate for both historical research and eventual execution integration if the user selects Zerodha.

### Upstox
- Historical and intraday candle APIs support configurable minute intervals.
- Current documentation also exposes India VIX and global index/indicator instruments through the broader API ecosystem.
- Some fine-resolution historical intervals have shorter documented retention than the multi-year target, so it is best treated as a useful complementary source unless the exact required history is confirmed.

### Yahoo Finance / yfinance
- Keep only as a prototyping fallback.
- It has been sufficient for 60-day experiments but should not be considered the research-quality source for the final model.

## New information layers to add

1. Multi-year 1m/5m NSE equity history.
2. NIFTY 50 and BANK NIFTY intraday candles.
3. India VIX.
4. Sector index intraday history.
5. Previous-day and overnight gap context.
6. Time-of-day relative volume and turnover.
7. Cross-sectional breadth and sector breadth.
8. Stock beta / residual return versus market and sector.
9. Futures data and OI where available.
10. Options context such as ATM IV / OI / PCR only when a reliable historical source is available.
11. Bid/ask, depth, order-book imbalance, or tick data only from a source whose licensing and history are suitable.

## Research rules

- No source is trusted merely because it is available.
- Record provider, interval, timezone, adjustment method, corporate-action handling and missing-data policy with each dataset.
- Avoid mixing differently adjusted price histories without explicit normalization.
- Training labels must remain executable: signal at t, entry at the next tradable price, exit at a future tradable price.
- Preserve chronological and cross-stock out-of-sample validation.
- Include statutory/broker costs and slippage sensitivity before approving any edge.
- No live trading is enabled by v29A.

## Recommended sequence

1. Run `run_data_capability_audit_v29a.py`.
2. Choose the preferred historical provider based on available account/API credentials.
3. Build a provider-specific downloader and symbol-master mapper.
4. Backfill 1-3 years of 5-minute data into a local canonical store.
5. Add market/sector/VIX/derivative context.
6. Run data-quality checks before any new model is trained.
7. Only then proceed to v29B opportunity meta-model research.
