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
import re
import requests

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
NEWSAPI_TIMEOUT = 8  # seconds

# Legal-entity suffixes that yfinance's `longName` includes but real news
# headlines almost never spell out verbatim (an article says "Ford", not
# "Ford Motor Company"; "Coca-Cola", not "The Coca-Cola Company"). Stripped
# before building the exact-phrase search below — see _search_phrase().
_SUFFIX_RE = re.compile(
    r"[,]?\s+(Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|Limited|PLC|"
    r"Holdings|Motor Company|Group)\s*$",
    re.IGNORECASE,
)

# Small, evidence-based denylist (NOT a broad allowlist — an allowlist of
# financial-news domains was tried first and reverted: it filtered out
# clearly relevant results from legitimate sources not on the list, such as
# PRNewswire, GlobeNewswire and trade press like WWD). Exact-phrase qInTitle
# matching (below) already fixes the "matches a random unrelated company"
# failure mode. What it can't fix is a short company name that is ALSO a
# common word/brand outside finance: "Ford" is used constantly in
# classic-car enthusiast headlines, unrelated to Ford stock. Each domain
# below was observed during manual testing to produce exclusively this kind
# of off-topic-but-title-matching result, never a genuine business/product
# article — see Diario Tecnico for the specific examples (e.g. "1974 Ford
# Bronco Sport 302" on bringatrailer.com).
EXCLUDED_NOISE_DOMAINS = ",".join([
    "bringatrailer.com",   # classic/collector car auction listings
    "slickdeals.net",      # consumer deals/coupons, not company news
    "fark.com",            # link-aggregator/forum, not original reporting
    "freerepublic.com",    # forum, not original reporting
])


def _search_phrase(name: str) -> str:
    """
    Turn a yfinance company name (e.g. "Ford Motor Company") into the short
    form actually used in news headlines (e.g. "Ford"), by stripping trailing
    legal-entity suffixes and a leading "The ". Applied in a loop since a
    name can have more than one trailing clause to strip (e.g. "X Inc.,
    a Delaware Corporation" — not expected from yfinance in practice, but
    cheap to handle defensively).
    """
    prev = None
    while prev != name:
        prev = name
        name = _SUFFIX_RE.sub("", name).strip()
    if name.lower().startswith("the "):
        name = name[4:].strip()
    return name


def get_news_for_company(company_name: str, ticker: str, max_articles: int = 3) -> list[dict]:
    """
    Fetch recent news headlines mentioning the given company.

    Searches by company name (falls back to the ticker symbol if
    company_name is empty/None), since NewsAPI's full-text search generally
    returns more relevant results for a company name than for a bare ticker
    symbol (e.g. "Apple Inc." vs "AAPL").

    Precision measures keep results on-topic. A plain q=<company name>
    search against the "everything" endpoint, unquoted, matches each word in
    the name independently, anywhere in the article's full body. For a
    single common word like "Apple" this pulls in unrelated results
    (recipes, fruit-growing). For multi-word legal names like "Tesla, Inc."
    or "Ford Motor Company" it is much worse: words like "Inc" or "Company"
    are common enough in *any* corporate press release that a search for
    "Tesla, Inc." returned wire-service articles about entirely unrelated
    companies (confirmed in manual testing — see Diario Tecnico).

    Current approach: `qInTitle` with the company name reduced to its short,
    headline-form name (see _search_phrase — "Ford", not "Ford Motor
    Company") and wrapped in double quotes for an EXACT PHRASE match, plus
    `excludeDomains` to drop a small set of sources observed to produce
    off-topic name-collision results (see EXCLUDED_NOISE_DOMAINS above).

    Returns a list of dicts: {"title", "source", "url", "published_at"}.
    Never raises — returns an empty list on any failure (missing API key,
    network error, rate limit, malformed response, no results), so a news
    outage never blocks stock data retrieval or the LLM response.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return []

    raw_query = (company_name or ticker or "").strip()
    if not raw_query:
        return []

    query = _search_phrase(raw_query) or raw_query

    params = {
        "qInTitle": f'"{query}"',
        "excludeDomains": EXCLUDED_NOISE_DOMAINS,
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
