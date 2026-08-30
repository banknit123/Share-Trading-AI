# Market Structure and Regimes

## Why regime matters

A trading signal can behave differently in a trending, ranging, volatile or quiet market. The model should therefore classify context before interpreting an entry signal.

## Candidate regime dimensions

### Trend state
Possible measurable definitions:
- price relative to EMA20 / EMA50
- EMA slope over recent bars
- higher-high / higher-low persistence
- rolling linear-regression slope
- directional movement / trend-strength measures

Suggested states:
- strong uptrend
- weak uptrend
- sideways
- weak downtrend
- strong downtrend

### Volatility state
Possible measures:
- ATR / price
- rolling realised volatility
- current true range relative to historical median
- opening-range width

Suggested states:
- compressed
- normal
- expanded
- extreme

### Volume/liquidity state
Possible measures:
- current volume / rolling median volume
- volume z-score
- turnover
- spread/slippage proxy when available

Suggested states:
- low participation
- normal participation
- volume expansion

### Cross-sectional market state
For the 10-stock NSE universe:
- proportion above EMA20
- median 5m/30m/60m return
- dispersion of stock returns
- breadth: advancing vs declining names
- concentration: whether one or two names dominate the move

### Time-of-day state
Do not assume opening, middle and late sessions have equal edge. Measure separately.

Candidate buckets:
- opening discovery
- post-open trend development
- midday compression
- afternoon continuation/reversal
- pre-close square-off

Exact boundaries must be learned from NSE evidence rather than fixed permanently.

## Regime-first decision rule

A signal should be evaluated conditionally:

`signal quality = f(stock state, market regime, volatility regime, volume regime, time of day)`

A breakout in a high-volume expanding trend is a different hypothesis from the same breakout in a quiet sideways market.

## Research requirement

Every future signal study should report results by regime and should not pool all environments together unless performance is demonstrably stable across them.
