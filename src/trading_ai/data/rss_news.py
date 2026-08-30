from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


@dataclass
class RSSArticle:
    title: str
    url: str
    domain: str | None
    published_at: datetime | None
    source: str | None


def _parse_pubdate(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def search_google_news_rss(query: str, maxrecords: int = 100, timeout: float = 15.0) -> list[RSSArticle]:
    """Fetch Google News RSS search results without an API key.

    This is a best-effort research fallback, not a guaranteed or licensed
    low-latency market-news feed. Keep requests modest and cache results.
    """
    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en",
    }
    headers = {
        "User-Agent": "Share-Trading-AI research/0.1 (+https://github.com/banknit123/Share-Trading-AI)",
        "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
    }
    resp = requests.get(GOOGLE_NEWS_RSS, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    out: list[RSSArticle] = []
    for item in root.findall("./channel/item")[: max(1, int(maxrecords))]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = _parse_pubdate(item.findtext("pubDate"))

        source_name = None
        source_node = item.find("source")
        if source_node is not None and source_node.text:
            source_name = source_node.text.strip()

        domain = None
        if source_node is not None:
            source_url = source_node.attrib.get("url")
            if source_url:
                domain = urlparse(source_url).netloc or None
        if domain is None and link:
            domain = urlparse(link).netloc or None

        out.append(
            RSSArticle(
                title=title,
                url=link,
                domain=domain,
                published_at=pub,
                source=source_name,
            )
        )
    return out
