"""
news_data.py
News Retrieval layer — fetches recent news headlines for a company via
NewsAPI.org (https://newsapi.org).

Iteration 2: yfinance's bundled `Ticker.news` (used in Iteration 1) is
inconsistent in coverage and freshness. NewsAPI's `/v2/everything` endpoint
gives a dedicated, keyword-searchable news feed, so headlines here are
sourced from NewsAPI instead.

Uses the free "Developer" tier (https://newsapi.org/pricing): 100 requests/
day, articles up to 1 month old with a ~24h publication delay, development/
testing use only. That is a good fit for this project, since markers do not
run the code with live API keys.
"""

import os
import requests

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
NEWSAPI_TIMEOUT = 8  # seconds


def get_news_for_company(company_name: str, ticker: str, max_articles: int = 3) -> list[dict]:
    """
    Fetch recent news headlines mentioning the given company.

    Searches by company name (falls back to the ticker symbol if
    company_name is empty/None), since NewsAPI's full-text search generally
    returns more relevant results for a company name than for a bare ticker
    symbol (e.g. "Apple Inc." vs "AAPL").

    Returns a list of dicts: {"title", "source", "url", "published_at"}.
    Never raises — returns an empty list on any failure (missing API key,
    network error, rate limit, malformed response, no results), so a news
    outage never blocks stock data retrieval or the LLM response.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return []

    query = (company_name or ticker or "").strip()
    if not query:
        return []

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(NEWSAPI_BASE_URL, params=params, timeout=NEWSAPI_TIMEOUT)
        data = resp.json()
    except requests.RequestException:
        return []
    except ValueError:
        # Response body was not valid JSON
        return []

    if resp.status_code != 200 or data.get("status") != "ok":
        return []

    articles = data.get("articles", [])[:max_articles]
    return [
        {
            "title": a.get("title", "").strip(),
            "source": (a.get("source") or {}).get("name") or "Unknown",
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", ""),
        }
        for a in articles
        if a.get("title") and a.get("title") != "[Removed]"
    ]


def build_news_context(news_items: list[dict]) -> str:
    """
    Format NewsAPI news items into a text block for injection into the RAG
    prompt, in the same style as build_data_context/build_comparative_context
    in financial_data.py. Returns an empty string if there are no items, so
    callers can unconditionally append the result without extra branching.
    """
    if not news_items:
        return ""

    lines = "\n".join(
        f"  - {item['title']} ({item['source']}, {item['published_at'][:10]})"
        for item in news_items
    )
    return f"\nRecent news (via NewsAPI):\n{lines}\n"
