# Sources and Books

This file records the conceptual sources used to generate testable research hypotheses. It is not a substitute for the original books and does not reproduce copyrighted text.

## John J. Murphy — Technical Analysis of the Financial Markets

Relevant themes for this project:
- trend identification
- support and resistance
- moving averages
- momentum/oscillators
- volume confirmation
- chart structure
- intermarket/context thinking

Project translation:
- trend-state features
- breakout/pullback hypotheses
- volume-confirmation features
- multi-timeframe context

## Mark Douglas — The Disciplined Trader

Relevant themes:
- probabilistic thinking
- discipline around predefined rules
- avoiding emotional/forced decisions
- accepting uncertainty

Project translation:
- CASH / abstention as a valid action
- no mandatory trading frequency
- predefined risk controls
- no threshold loosening merely to generate trades

## Kathy Lien — Day Trading and Swing Trading the Currency Market

Relevant themes:
- multi-timeframe analysis
- volatility and session behaviour
- macro/event context
- trade/risk planning

Project translation:
- 5m/10m/30m/60m/120m horizon interaction
- time-of-day studies
- volatility/event context as conditioning variables

Note: the source market is FX, so any concept must be independently validated on NSE equities.

## William J. O'Neil — How to Make Money in Stocks

Relevant themes:
- relative strength
- price/volume confirmation
- leadership
- breakout quality
- market context

Project translation:
- cross-sectional ranking
- top-relative-strength stock features
- volume-confirmed breakout studies

Note: much of the original framework is not intraday-specific, so it must be adapted and tested rather than copied.

## Larry Harris — Trading and Exchanges

Relevant themes:
- market microstructure
- liquidity
- spreads
- order execution
- transaction costs

Project translation:
- explicit cost engine
- slippage/spread modelling
- turnover controls
- realistic execution assumptions

## David Aronson — Evidence-Based Technical Analysis

Relevant themes:
- statistical testing of technical rules
- data-snooping risk
- rejecting anecdotal chart-pattern claims

Project translation:
- hypothesis registry
- chronological validation
- multiple-testing awareness
- reject signals that fail after-cost evidence

## Ernest P. Chan — Algorithmic Trading / Machine Trading

Relevant themes:
- systematic strategy research
- backtesting
- risk and execution
- quantitative feature/model evaluation

Project translation:
- walk-forward research pipeline
- feature engineering as hypotheses
- practical strategy/economics testing

## Marcos López de Prado — Advances in Financial Machine Learning

Relevant themes:
- leakage-aware financial ML
- overlapping labels
- purging/embargo concepts
- meta-labeling/selective prediction
- bet sizing
- backtest overfitting

Project translation:
- strict outcome-timestamp guards
- abstention/meta-gating
- calibration before sizing
- stronger controls around repeated experimentation

## Perry Kaufman — Trading Systems and Methods

Relevant themes:
- systematic trend/momentum methods
- volatility adaptation
- robust system design

Project translation:
- regime-aware systematic features
- parameter robustness tests

## Adam Grimes — The Art and Science of Technical Analysis

Relevant themes:
- market structure
- momentum
- pullbacks
- breakouts
- separating repeatable tendencies from chart folklore

Project translation:
- explicit setup definitions
- regime-conditioned signal studies

## Source hierarchy for the project

1. Our leakage-safe NSE out-of-sample evidence
2. Exchange/broker primary documentation for costs and execution rules
3. Quantitative/statistical literature
4. Established trading books as hypothesis generators
5. General internet/community claims only as low-priority ideas to test

No source is allowed to override observed negative after-cost evidence in our target market.
