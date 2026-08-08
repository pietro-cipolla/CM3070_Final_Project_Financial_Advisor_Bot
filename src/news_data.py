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
testing use only.
"""

import os
import re
import requests
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
NEWSAPI_TIMEOUT = 8  # seconds

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_CANDIDATE_MULTIPLIER = 3

_SUFFIX_RE = re.compile(
    r"[,]?\s+(Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|Limited|PLC|"
    r"Holdings|Motor Company|Group)\s*$",
    re.IGNORECASE,
)

EXCLUDED_NOISE_DOMAINS = ",".join([
    "bringatrailer.com",   
    "slickdeals.net",      
    "fark.com",            
    "freerepublic.com",    
])


def _search_phrase(name: str) -> str:
    """
    Turn a yfinance company name (e.g. "Ford Motor Company") into the short
    form actually used in news headlines (e.g. "Ford"), by stripping trailing
    legal-entity suffixes and a leading "The ". Applied in a loop since a
    name can have more than one trailing clause to strip (e.g. "X Inc.,
    a Delaware Corporation").
    """
    prev = None
    while prev != name:
        prev = name
        name = _SUFFIX_RE.sub("", name).strip()
    if name.lower().startswith("the "):
        name = name[4:].strip()
    return name


def _filter_relevant_articles(articles: list[dict], company_label: str) -> list[dict]:
    """
    Narrow a list of candidate articles (already title-matched by NewsAPI)
    down to the ones actually ABOUT the company as a business, not just
    ones where the company name happens to appear in the headline.

    Why this exists: domain-based filtering (allowlist, then denylist) hits a wall that no
    list of sites can fix, a short company name can collide with an
    unrelated everyday use of the same word on ANY site.
    That's a problem about the article's TOPIC, not its SOURCE, so this
    filters on content instead: one LLM call judges the whole candidate
    batch at once.

    Fails open, not closed: if the classification call itself fails for any
    reason, this returns the candidates unfiltered rather than dropping
    them, since showing untrimmed-but-fetched results is strictly better
    than showing none due to an unrelated API hiccup, consistent with the
    "a news outage never blocks the rest of the response" design used
    throughout this module.
    """
    if not articles:
        return articles

    numbered = "\n".join(f"{i+1}. {a['title']}" for i, a in enumerate(articles))
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=30,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Below is a numbered list of news headlines that matched a "
                        "text search for the company name, but the search cannot tell "
                        "whether each headline is actually about that company as a "
                        "business (its stock, products, earnings, leadership, deals, "
                        "controversies, etc.) versus an unrelated use of the same word "
                        "or a different entity that happens to share the name. Exclude, "
                        "in particular: a classic-car listing for a brand name; an "
                        "unrelated award whose acronym matches the company name; a "
                        "different person or place with the same name; and a headline "
                        "about something else entirely (sports, entertainment, unrelated "
                        "products, etc.) where the company name only appears as an "
                        "incidental sponsor/advertiser credit line, such as '... "
                        "Presented By Your Local Ford Dealers' on a sports article — "
                        "that is an ad tag, not news about the company. "
                        "Reply with ONLY a comma-separated list of the item numbers "
                        "that ARE genuinely about the company as a business, in the "
                        "original order, nothing else. If none qualify, reply NONE."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Company: {company_label}\n\nHeadlines:\n{numbered}",
                },
            ],
        )
        result = response.choices[0].message.content.strip().upper()
        if result == "NONE" or not result:
            return []
        keep_indices = set()
        for tok in result.split(","):
            tok = tok.strip()
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(articles):
                    keep_indices.add(idx)
        # Fail open on a malformed/empty reply rather than silently returning nothing.
        if not keep_indices and result != "NONE":
            return articles
        return [a for i, a in enumerate(articles) if i in keep_indices]
    except Exception:
        return articles


def get_news_for_company(company_name: str, ticker: str, max_articles: int = 3) -> list[dict]:
    """
    Fetch recent news headlines mentioning the given company.

    Searches by company name (falls back to the ticker symbol if
    company_name is empty/None), since NewsAPI's full-text search generally
    returns more relevant results for a company name than for a bare ticker
    symbol (e.g. "Apple Inc." vs "AAPL").

    Precision measures keep results on-topic. Two earlier attempts were
    tried and found insufficient during manual testing:
      - Plain q=<company name>, unquoted: matches each word in the name
        independently, anywhere in the article's full body. For a single
        common word like "Apple" this pulls in unrelated results (recipes,
        fruit-growing). For multi-word legal names like "Tesla, Inc." or
        "Ford Motor Company" it is much worse: words like "Inc" or "Company"
        are common enough in *any* corporate press release that a search
        for "Tesla, Inc." returned wire-service articles about entirely
        unrelated companies.
      - Adding a fixed allowlist of financial-news domains on top of
        `qInTitle`: this filtered out clearly relevant, on-topic results
        from legitimate sources not on the list (PRNewswire, GlobeNewswire,
        trade press like WWD), too aggressive, discarding good results
        along with the bad.
    Current approach: `qInTitle` with the company name reduced to its short,
    headline-form name and wrapped in double quotes for an EXACT PHRASE match. This
    requires the literal short name to appear in the headline, which is
    both narrow enough to exclude generic-word false positives and broad
    enough to keep results from any legitimate outlet NewsAPI indexes.

    This alone does not (and structurally cannot, without a dedicated
    financial-news API or real entity disambiguation) solve every case: a
    short company name can coincide with an unrelated common use of the
    same word, "Ford" in classic-car enthusiast headlines, "Coty" as a
    "Coach/Citizen Of The Year" award acronym. Two more layers handle that,
    applied in order: `excludeDomains`
    trims specific low-editorial-quality sources observed to produce this
    kind of noise, cheaply and with no API cost; then _filter_relevant_
    articles() makes one LLM call over the surviving candidates to judge
    actual topical relevance, which is what generalizes to sources and
    name collisions not seen during testing (domain lists, by definition,
    only ever cover cases already observed).

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
    company_label = raw_query  # full name for the relevance-filter prompt

    params = {
        "qInTitle": f'"{query}"',
        "excludeDomains": EXCLUDED_NOISE_DOMAINS,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles * _CANDIDATE_MULTIPLIER,
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

    candidates = [
        {
            "title": a.get("title", "").strip(),
            "source": (a.get("source") or {}).get("name") or "Unknown",
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", ""),
        }
        for a in data.get("articles", [])
        if a.get("title") and a.get("title") != "[Removed]"
    ]
    if not candidates:
        return []

    relevant = _filter_relevant_articles(candidates, company_label)
    return relevant[:max_articles]


def build_news_context(news_items: list[dict], include_sentiment: bool = True) -> str:
    """
    Format NewsAPI news items into a text block for injection into the RAG
    prompt, in the same style as build_data_context/build_comparative_context
    in financial_data.py. Returns an empty string if there are no items, so
    callers can unconditionally append the result without extra branching.

    Each headline is optionally tagged with its VADER
    sentiment label, and an aggregate "overall news sentiment" line is
    added, so the LLM (and, via render_news in app.py, the user) has an
    explicit signal alongside the raw headlines rather than having to
    infer tone from the text itself. include_sentiment defaults to True
    but can be disabled by callers that only want the plain headline list
    (e.g. existing Iteration 2 tests that check exact substrings).
    """
    if not news_items:
        return ""

    items = score_news_sentiment(news_items) if include_sentiment else news_items

    lines = "\n".join(
        f"  - {item['title']} ({item['source']}, {item['published_at'][:10]})"
        + (f" [sentiment: {item['sentiment_label']}]" if include_sentiment else "")
        for item in items
    )

    header = "\nRecent news (via NewsAPI):\n"
    if include_sentiment:
        summary = summarize_sentiment(items)
        header = (
            f"\nRecent news (via NewsAPI) — overall sentiment: {summary['overall_label']} "
            f"({summary['positive']} positive, {summary['neutral']} neutral, "
            f"{summary['negative']} negative):\n"
        )

    return f"{header}{lines}\n"


# Sentiment analysis
_sentiment_analyzer = SentimentIntensityAnalyzer()

# VADER's own documented thresholds for classifying the compound score.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def score_headline_sentiment(title: str) -> dict:
    """
    Score a single headline's sentiment using VADER (vaderSentiment), a
    lexicon- and rule-based sentiment analyzer well-suited to short,
    informal text like news headlines.

    Returns {"compound": float in [-1, 1], "label": "positive"|"neutral"|"negative"}.
    An empty/missing title scores neutral (VADER returns compound 0.0 for
    empty text), rather than raising.
    """
    scores = _sentiment_analyzer.polarity_scores(title or "")
    compound = scores["compound"]
    if compound >= POSITIVE_THRESHOLD:
        label = "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    return {"compound": compound, "label": label}


def score_news_sentiment(news_items: list[dict]) -> list[dict]:
    """
    Enrich a list of news items (as returned by get_news_for_company) with
    per-headline sentiment. Returns NEW dicts (does not mutate the input
    list or its items), each with "sentiment_compound" and
    "sentiment_label" added alongside the existing title/source/url/
    published_at fields.
    """
    enriched = []
    for item in news_items:
        sentiment = score_headline_sentiment(item.get("title", ""))
        enriched.append({
            **item,
            "sentiment_compound": sentiment["compound"],
            "sentiment_label": sentiment["label"],
        })
    return enriched


def summarize_sentiment(scored_items: list[dict]) -> dict:
    """
    Aggregate an overall sentiment summary across a list of already
    sentiment-scored news items (as returned by score_news_sentiment —
    each item must already carry "sentiment_label" and
    "sentiment_compound").

    Returns {"positive": n, "neutral": n, "negative": n,
    "overall_label": str, "average_compound": float or None}.
    overall_label is the majority label among the three counts; a tie
    (including a 3-way tie) resolves to "neutral", a deliberately
    cautious default rather than guessing a direction from an ambiguous
    split. An empty input returns all-zero counts, overall_label
    "neutral", and average_compound None, rather than raising or silently
    reporting a fabricated non-zero value.
    """
    if not scored_items:
        return {"positive": 0, "neutral": 0, "negative": 0, "overall_label": "neutral", "average_compound": None}

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for item in scored_items:
        label = item.get("sentiment_label", "neutral")
        counts[label] = counts.get(label, 0) + 1

    average_compound = sum(item.get("sentiment_compound", 0.0) for item in scored_items) / len(scored_items)

    max_count = max(counts.values())
    top_labels = [label for label, c in counts.items() if c == max_count]
    overall_label = "neutral" if len(top_labels) > 1 else top_labels[0]

    return {**counts, "overall_label": overall_label, "average_compound": average_compound}
