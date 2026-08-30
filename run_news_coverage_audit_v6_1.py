from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import time

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.gdelt_news import search_articles
from trading_ai.data.rss_news import search_google_news_rss


QUERY_MAP = {
    "RELIANCE.NS": '"Reliance Industries" India',
    "HDFCBANK.NS": '"HDFC Bank" India',
    "ICICIBANK.NS": '"ICICI Bank" India',
    "INFY.NS": 'Infosys India',
    "TCS.NS": '"Tata Consultancy Services" India',
    "SBIN.NS": '"State Bank of India"',
    "BHARTIARTL.NS": '"Bharti Airtel" India',
    "ITC.NS": '"ITC Limited" India',
    "LT.NS": '"Larsen & Toubro" India',
    "AXISBANK.NS": '"Axis Bank" India',
}


def _fmt_dt(value: datetime | None) -> str:
    if value is None:
        return "n/a"
    return value.astimezone(timezone.utc).isoformat()


def main() -> None:
    print("Share-Trading-AI v6.1 resilient news coverage audit")
    print("No trades are placed. Sources: GDELT primary + Google News RSS fallback; no API key required.")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)

    total = 0
    gdelt_ok = 0
    rss_ok = 0
    failures = 0

    for symbol in DEFAULT_CONFIG.universe:
        query = QUERY_MAP[symbol]
        provider = None
        count = 0
        latest = None
        domains: Counter[str] = Counter()

        # First try GDELT with a smaller query size and one short retry.
        gdelt_error = None
        for attempt in range(2):
            try:
                articles = search_articles(query, timespan="7d", maxrecords=40)
                if articles:
                    provider = "GDELT"
                    count = len(articles)
                    latest = max((a.seen_at for a in articles if a.seen_at is not None), default=None)
                    domains.update(a.domain for a in articles if a.domain)
                    gdelt_ok += 1
                    break
            except Exception as exc:
                gdelt_error = exc
                if attempt == 0:
                    time.sleep(1.0)

        # Fall back to RSS if GDELT failed or returned nothing.
        if provider is None:
            try:
                articles = search_google_news_rss(query, maxrecords=60, timeout=15.0)
                if articles:
                    provider = "RSS"
                    count = len(articles)
                    latest = max((a.published_at for a in articles if a.published_at is not None), default=None)
                    domains.update(a.domain for a in articles if a.domain)
                    rss_ok += 1
            except Exception as exc:
                if gdelt_error is None:
                    gdelt_error = exc

        if provider is None:
            failures += 1
            print(f"{symbol:15} ERROR: no usable news source ({gdelt_error})")
            continue

        total += count
        top = ", ".join(f"{d}({n})" for d, n in domains.most_common(5)) or "n/a"
        print(f"{symbol:15} source={provider:5} articles={count:3d} latest={_fmt_dt(latest)}")
        print(f"  top domains: {top}")

    print("\nSource-health summary")
    print(f"GDELT successful symbols: {gdelt_ok}/{len(DEFAULT_CONFIG.universe)}")
    print(f"RSS fallback successful symbols: {rss_ok}/{len(DEFAULT_CONFIG.universe)}")
    print(f"No-source failures: {failures}/{len(DEFAULT_CONFIG.universe)}")
    print(f"Total articles returned: {total}")

    usable = failures <= 2 and total >= 150
    print("Coverage gate:", "PASS FOR NEWS-FEATURE PROTOTYPE" if usable else "FAIL - NEWS SOURCE LAYER NEEDS MORE WORK")
    print("This is a research-data availability check only; live trading remains disabled.")


if __name__ == "__main__":
    main()
