# Research Protocol for New Trading Ideas

Use this protocol before adding any indicator, setup, rule or book-derived idea to the trading engine.

## 1. State the hypothesis

Example:

`Volume-confirmed 30m breakouts in top-quartile relative-strength stocks have positive 60m after-cost expectancy.`

Avoid vague statements such as `breakouts work`.

## 2. Define measurable inputs

Specify exact features and calculations without using future data.

## 3. Define target and execution

Specify:
- prediction horizon
- signal timestamp
- executable entry timestamp/price
- exit rule
- intraday square-off rule
- cost model

## 4. Define context

Test whether the effect depends on:
- stock
- horizon
- market regime
- volatility regime
- volume regime
- time of day

## 5. Predeclare search space

Record thresholds/model variants before examining final test outcomes where practical.

## 6. Leakage-safe training

A row may enter training only when its future target outcome was known before the calibration/test cutoff.

## 7. Chronological validation

Minimum preferred sequence:
- model training
- gate-training
- later gate-validation
- unseen walk-forward test

## 8. Evaluate after costs

Report both gross and realistic net results. A signal with positive gross but negative net expectancy is not tradeable in its current form.

## 9. Evaluate stability

Require evidence across:
- multiple dates
- sufficient observations
- more than one chronological block

Investigate whether results are driven by one stock, one date or one unusual market regime.

## 10. Decide

Allowed research decisions:
- `KEEP` — robust positive after-cost evidence
- `CONDITIONAL` — positive only in a defined stock/horizon/regime
- `RESEARCH` — promising but insufficient evidence
- `REJECT` — no repeatable edge

## 11. Protect the final holdout

Repeatedly redesigning against the same test period converts it into development data. Once used to make design decisions, it is no longer an untouched test.

## 12. Live/paper promotion gate

A research idea should not be promoted toward broker execution until it has:
- positive walk-forward after-cost expectancy
- acceptable drawdown
- adequate observation count
- stable performance across time
- realistic execution assumptions
- explicit failure/abstention behaviour

The default state remains research/paper trading, not live orders.
