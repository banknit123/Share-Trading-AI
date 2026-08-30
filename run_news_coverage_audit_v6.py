from __future__ import annotations

from collections import Counter
from statistics import mean

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.gdelt_news import search_articles


COMPANY_QUERIES = {
    "RELIANCE.NS": '"Reliance Industries"',
    "HDFCBANK.NS": '"HDFC Bank"',
    "ICICIBANK.NS": '"ICICI Bank"',
    "INFY.NS": 'Infosys',
    "TCS.NS": '"Tata Consultancy Services"',
    "SBIN.NS": '"State Bank of India"',
    "BHARTIARTL.NS": '"Bharti Airtel"',
    "ITC.NS": '"ITC Limited"',
    "LT.NS": '"Larsen & Toubro"',
    "AXISBANK.NS": '"Axis Bank"',
}


def main() -> None:
    print("Share-Trading-AI v6 news/event coverage audit")
    print("No trades are placed. Source: GDELT DOC public API; no API key required.")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)
    print()

    total = 0
    for symbol in DEFAULT_CONFIG.universe:
        query = COMPANY_QUERIES.get(symbol, symbol.replace(".NS", ""))
        try:
            articles = search_articles(query, timespan="7d", maxrecords=100)
        except Exception as exc:
            print(f"{symbol:15} ERROR: {exc}")
            continue

        total += len(articles)
        tones = [a.tone for a in articles if a.tone is not None]
        domains = Counter(a.domain for a in articles if a.domain)
        latest = max((a.seen_at for a in articles if a.seen_at), default=None)
        avg_tone = mean(tones) if tones else 0.0
        print(
            f"{symbol:15} articles={len(articles):3d} "
            f"avg_tone={avg_tone:7.3f} latest={latest.isoformat() if latest else 'n/a'}"
        )
        if domains:
            print("  top domains:", ", ".join(f"{d}({n})" for d, n in domains.most_common(5)))

    print(f"\nTotal articles returned across universe: {total}")
    print("Next gate: only build event features if coverage is broad and timestamp quality is usable.")


if __name__ == "__main__":
    main()
