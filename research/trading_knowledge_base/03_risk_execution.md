# Risk, Trade Economics and Execution

## Core principle

A predicted price rise is not enough. The trade should be evaluated on expected net outcome after all realistic costs and execution assumptions.

## Pre-trade economics

For every candidate trade estimate:
- expected gross return
- brokerage
- exchange transaction charges
- STT
- SEBI turnover fees
- stamp duty
- GST
- expected slippage
- optional safety margin for model error

Then calculate:

`expected_net_return = expected_gross_return - expected_total_cost_rate - safety_margin`

A trade is eligible only if expected net return is positive and the strategy/regime has demonstrated positive out-of-sample after-cost expectancy.

## Abstention

The engine may scan continuously while taking zero trades. No-trade is a valid outcome when:
- no robust gate exists
- opportunity quality is insufficient
- expected net edge is too small
- risk budget is exhausted
- liquidity/execution conditions are poor
- selected horizon would extend beyond intraday square-off

## Position sizing

Sizing should depend on evidence and risk rather than a fixed percentage chosen by convenience.

Candidate inputs:
- confidence / calibrated probability
- expected net edge
- forecast uncertainty
- volatility / ATR
- stop distance
- current portfolio exposure
- correlation / concentration among open positions
- remaining daily risk budget

Research first; do not use Kelly sizing without robust probability calibration.

## Turnover control

Earlier project results showed excessive turnover can overwhelm small gross edges. Controls to test:
- minimum expected net edge
- minimum time between re-entry in same symbol
- no resizing unless expected benefit exceeds additional costs
- maximum concurrent positions
- learned opportunity budget by session
- rank persistence requirement before rotation

Trade limits are maxima, not targets.

## Exit logic

Candidate exit types to test separately:
- planned horizon exit
- short-horizon forecast reversal
- loss of relative-strength rank
- trailing profit protection
- volatility-adjusted stop
- time stop
- end-of-day mandatory square-off

Avoid introducing several exit rules simultaneously without attribution testing.

## Daily risk controls

Before any live deployment, test:
- maximum daily loss
- maximum drawdown stop
- maximum gross exposure
- maximum symbol exposure
- consecutive-loss pause
- abnormal-volatility halt
- stale/missing-data halt

These are capital-protection constraints, not sources of predictive edge.

## Execution realism

Backtests should use next-bar execution or another realistic fill model. Never execute at a price that was only known after the signal was formed.

Where possible, future work should include bid/ask spread and slippage rather than relying only on OHLC bars.
