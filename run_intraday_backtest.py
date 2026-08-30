from trading_ai.backtest.intraday import intraday_backtest
from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.market_data import download_history
from trading_ai.features.intraday import HORIZON_BARS, add_intraday_features


def main() -> None:
    print("Share-Trading-AI intraday research backtest")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)
    print("Data: 5-minute candles, research source only")

    for symbol in DEFAULT_CONFIG.universe:
        try:
            raw = download_history(symbol, period="60d", interval="5m")
            featured = add_intraday_features(raw)
            print(f"\n{symbol}")
            for horizon in HORIZON_BARS:
                result = intraday_backtest(featured, horizon=horizon)
                print(
                    f"  {horizon:3} trades={result.trades:4d} "
                    f"strategy={result.strategy_return:8.2%} "
                    f"buy_hold={result.buy_hold_return:8.2%} "
                    f"win_rate={result.win_rate:7.2%} "
                    f"max_dd={result.max_drawdown:7.2%}"
                )
        except Exception as exc:
            print(f"\n{symbol} ERROR: {exc}")


if __name__ == "__main__":
    main()
