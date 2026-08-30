from trading_ai.backtest.engine import walk_forward_backtest
from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.market_data import download_history
from trading_ai.features.technical import add_features
from trading_ai.validation.readiness import evaluate_readiness


def main() -> None:
    print("Share-Trading-AI research backtest")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)
    results = []
    for symbol in DEFAULT_CONFIG.universe:
        try:
            raw = download_history(symbol, period="1y", interval="1d")
            featured = add_features(raw)
            result = walk_forward_backtest(featured, DEFAULT_CONFIG)
            results.append(result)
            print(
                f"{symbol:15} trades={result.trades:3d} "
                f"strategy={result.total_return:8.2%} "
                f"buy_hold={result.buy_hold_return:8.2%} "
                f"win_rate={result.win_rate:7.2%} "
                f"max_dd={result.max_drawdown:7.2%}"
            )
        except Exception as exc:
            print(f"{symbol:15} ERROR: {exc}")

    report = evaluate_readiness(results)
    print("\nResearch readiness gate")
    print(f"Profitable symbols: {report.profitable_symbols}/{report.tested_symbols}")
    print(f"Average strategy return: {report.average_return:.2%}")
    print(f"Median win rate: {report.median_win_rate:.2%}")
    print("Status:", "PASS" if report.passed else "FAIL - PAPER/RESEARCH ONLY")
    for reason in report.reasons:
        print(" -", reason)


if __name__ == "__main__":
    main()
