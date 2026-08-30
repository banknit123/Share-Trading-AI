# Signal Library

This file translates common technical-analysis ideas into measurable features. Inclusion here does not mean the signal is profitable.

## Trend and momentum

Candidate features:
- 5m / 10m / 30m / 60m / 120m returns
- EMA20 distance
- EMA50 distance
- EMA20 slope
- EMA20-EMA50 spread
- rolling linear-regression slope
- consecutive higher closes / lower closes
- rate of change
- RSI or oscillator state as a secondary feature, not a standalone rule

Hypotheses:
- aligned short/medium trend may improve continuation probability
- momentum may work better when volume and market breadth confirm
- overextended momentum may reverse rather than continue

## Breakout / breakdown

Candidate features:
- close above recent N-bar high
- close below recent N-bar low
- breakout distance / ATR
- opening-range breakout
- breakout volume z-score
- close location within candle
- wick/body proportions

Hypotheses:
- breakouts with abnormal volume and strong relative strength may outperform unconfirmed breakouts
- late breakouts after excessive extension may underperform

## Pullback continuation

Candidate features:
- trend state positive
- short-term negative return against longer-term positive trend
- pullback depth relative to ATR
- distance to EMA/VWAP
- volume contraction during pullback
- renewed positive candle/volume confirmation

## Mean reversion / reversal

Candidate features:
- distance from VWAP
- distance from rolling mean / EMA
- RSI/oscillator extreme
- long wick / rejection candle
- volatility spike
- reversal after failed breakout

Hypothesis must be conditioned on regime; mean reversion in a strong trend can be hazardous.

## Relative strength

Candidate features:
- stock return minus universe median return
- rolling rank among the 10 stocks
- persistence of top-quartile rank
- relative strength across multiple horizons

Cross-sectional ranking is especially relevant because earlier project research found more promise in relative 120m ranking than in absolute direction prediction.

## Volume confirmation

Candidate features:
- volume / rolling median
- volume z-score
- price move per unit volume
- volume expansion with breakout
- volume contraction during pullback

## Candle structure

Candidate features:
- candle body / range
- upper wick / range
- lower wick / range
- close location value
- gap from prior close
- range expansion
- inside/outside bar state
- multi-candle persistence

## Multi-horizon agreement

The engine currently evaluates 5m, 10m, 30m, 60m and 120m. Agreement should not automatically imply a trade; it is a candidate confidence feature to be validated.

Useful measurements:
- number of positive horizon forecasts
- weighted forecast sign agreement
- dispersion between horizon forecasts
- whether short horizons confirm or contradict the selected holding horizon

## Time-of-day features

Candidate features:
- minutes since open
- opening-range location
- intraday high/low proximity
- midday compression
- late-session reversal/continuation state

These should be learned empirically for NSE rather than imposed from generic trading lore.
