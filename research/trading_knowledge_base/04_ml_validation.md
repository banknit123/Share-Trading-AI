# Machine-Learning Validation Rules

## Objective

Prevent false profitability caused by leakage, repeated tuning and backtest overfitting.

## Time ordering

For a replay date D:
- training examples may be used only when their target outcome was known before D
- threshold calibration must use periods before the final validation/test block
- the replay/test day must not influence model parameters, thresholds, session limits or feature selection

## Walk-forward design

Preferred structure:
1. expanding or rolling training window
2. chronological gate-training block
3. later validation block
4. next unseen trading day
5. advance one day and repeat

This is the basis of the v24/v24.1 abstention framework.

## Purging and embargo

Overlapping forward-return labels can leak information across boundaries. For each horizon, exclude observations whose outcome timestamp crosses into the validation/test period. Consider purging adjacent observations when labels overlap heavily.

## Multiple-testing control

Every new indicator, threshold and regime split increases the probability of finding a lucky historical pattern.

Required safeguards:
- predeclare candidate feature/threshold grids where practical
- keep search spaces compact
- report number of hypotheses tested
- demand stability across chronological blocks
- maintain a final holdout period that is not repeatedly redesigned against
- prefer simpler rules when performance is similar

## Metrics

Prediction metrics:
- rank IC / Spearman correlation
- direction accuracy
- calibration error
- top-k hit rate

Trading metrics:
- mean and median gross return/trade
- mean and median net return/trade
- win rate after costs
- profit factor
- average trades/day
- turnover
- positive-day rate
- cumulative return
- maximum drawdown
- worst session
- sensitivity to higher costs/slippage

A high classifier accuracy without positive net expectancy is not sufficient.

## Calibration

Predicted returns should be tested for calibration. For example, signals predicting +0.5% should not systematically realise +0.05%.

Useful studies:
- predicted-return deciles vs realised returns
- confidence buckets
- stock-specific calibration
- horizon-specific calibration
- regime-specific calibration

## Abstention / selective prediction

The model should have a CASH state. Evaluate:
- coverage: percentage of scans/days where model trades
- selective accuracy: accuracy only on accepted signals
- selective net expectancy
- capital preserved on abstained days

Do not increase coverage merely to increase trade count.

## Model comparison

A more complex model should replace a simpler model only if it improves out-of-sample after-cost results consistently, not just AUC or in-sample fit.

## Reproducibility

Every experiment should record:
- code version / commit
- universe
- date range
- features
- model parameters
- costs
- execution assumption
- thresholds
- results
- decision: keep / conditional / reject
