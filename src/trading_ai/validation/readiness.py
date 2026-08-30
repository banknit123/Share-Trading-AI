from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ReadinessReport:
    passed: bool
    profitable_symbols: int
    tested_symbols: int
    average_return: float
    median_win_rate: float
    reasons: list[str]


def evaluate_readiness(results: Iterable[object]) -> ReadinessReport:
    rows = list(results)
    if not rows:
        return ReadinessReport(False, 0, 0, 0.0, 0.0, ["No valid backtest results."])

    returns = [float(r.total_return) for r in rows]
    win_rates = sorted(float(r.win_rate) for r in rows)
    profitable = sum(r > 0 for r in returns)
    avg_return = sum(returns) / len(returns)
    midpoint = len(win_rates) // 2
    if len(win_rates) % 2:
        median_win_rate = win_rates[midpoint]
    else:
        median_win_rate = (win_rates[midpoint - 1] + win_rates[midpoint]) / 2

    reasons: list[str] = []
    if profitable < max(3, len(rows) // 2):
        reasons.append("Too few symbols are profitable out of sample.")
    if avg_return <= 0:
        reasons.append("Average strategy return is not positive.")
    if median_win_rate < 0.50:
        reasons.append("Median win rate is below 50%.")

    return ReadinessReport(
        passed=not reasons,
        profitable_symbols=profitable,
        tested_symbols=len(rows),
        average_return=avg_return,
        median_win_rate=median_win_rate,
        reasons=reasons,
    )
