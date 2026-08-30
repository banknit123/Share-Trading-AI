# Share-Trading-AI Research Knowledge Base

This folder converts established trading concepts into testable research hypotheses for the Share-Trading-AI project.

## Principle

No idea is accepted because it appears in a famous book. Every idea must pass leakage-safe NSE testing after realistic costs before it can influence live/paper trading decisions.

Pipeline:

1. Source concept
2. Translate to measurable features
3. Define falsifiable hypothesis
4. Backtest with strict anti-leakage rules
5. Measure gross and after-cost performance
6. Validate across multiple chronological blocks
7. Keep, condition, or reject
8. Only then consider use in the trading engine

## Current project lessons already established

- Frequent rebalancing can destroy a small gross edge through statutory and brokerage costs.
- Zero brokerage is not zero transaction cost.
- A single good day is not enough evidence of an edge.
- Generic rules applied to all stocks and horizons have not yet shown robust positive after-cost expectancy.
- Abstention is a valid trading decision. If no historically robust opportunity exists, the engine should remain in cash.
- Thresholds and trade limits should be learned from prior data, not copied from examples or chosen after seeing the test day.

## Files

- `01_market_structure_and_regimes.md` — trend, range, volatility and session-state concepts.
- `02_signal_library.md` — measurable technical and price/volume features.
- `03_risk_execution.md` — cost-aware entry, sizing, exits and turnover control.
- `04_ml_validation.md` — anti-leakage, walk-forward testing and model-selection rules.
- `05_sources_and_books.md` — source map and how each source contributes to the research program.
- `06_research_protocol.md` — mandatory workflow for adding a new trading idea.
- `strategy_hypotheses.csv` — hypothesis registry with testing status.

## Copyright note

The knowledge base contains original summaries and research abstractions. It does not reproduce copyrighted books or long passages from them.
