from collections import defaultdict
from statistics import mean, median

from trading_ai.backtest.intraday_optimizer import optimize_intraday_threshold
from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.market_data import download_history
from trading_ai.features.intraday import HORIZON_BARS, add_intraday_features


def main() -> None:
    print("Share-Trading-AI nested intraday optimisation")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)
    print("Split: 60% train / 20% validation / 20% untouched test")

    by_horizon = defaultdict(list)

    for symbol in DEFAULT_CONFIG.universe:
        try:
            raw = download_history(symbol, period="60d", interval="5m")
            featured = add_intraday_features(raw)
            print(f"\n{symbol}")
            for horizon in HORIZON_BARS:
                result = optimize_intraday_threshold(featured, horizon)
                by_horizon[horizon].append(result)
                print(
                    f"  {horizon:3} threshold={result.threshold:.2f} "
                    f"val_trades={result.validation.trades:3d} val_ret={result.validation.strategy_return:7.2%} "
                    f"test_trades={result.test.trades:3d} test_ret={result.test.strategy_return:7.2%} "
                    f"win={result.test.win_rate:6.2%} dd={result.test.max_drawdown:6.2%} "
                    f"buy_hold={result.test_buy_hold_return:7.2%}"
                )
        except Exception as exc:
            print(f"\n{symbol} ERROR: {exc}")

    print("\nCross-symbol untouched-test summary")
    for horizon in HORIZON_BARS:
        results = by_horizon.get(horizon, [])
        if not results:
            print(f"{horizon}: no results")
            continue
        returns = [r.test.strategy_return for r in results]
        wins = [r.test.win_rate for r in results if r.test.trades > 0]
        profitable = sum(r > 0 for r in returns)
        trades = sum(r.test.trades for r in results)
        print(
            f"{horizon}: profitable={profitable}/{len(results)} "
            f"avg_return={mean(returns):.2%} median_return={median(returns):.2%} "
            f"total_trades={trades} median_win={(median(wins) if wins else 0.0):.2%}"
        )

    print("\nThis optimisation is still research-only. Do not enable live trading based on these results alone.")


if __name__ == "__main__":
    main()
