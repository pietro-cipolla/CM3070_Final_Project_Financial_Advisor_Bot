"""
rag_pipeline.py
RAG Pipeline layer — query intent classification, ticker extraction and
prompt construction.

Iteration 1 additions (Preliminary Report, Table 4.2 — HIGH priority items):
  1. Multi-ticker extraction: a query can now reference up to 3 companies
     (e.g. "Compare Apple, Microsoft and Google"), instead of only the first
     ticker found.
  2. Query intent classification: queries are classified before ticker
     extraction runs, so open-ended / off-topic queries are routed to a
     clarification prompt instead of silently failing or hallucinating
     an answer with no financial grounding.
"""

import os
from openai import OpenAI
from src.financial_data import build_data_context, build_comparative_context

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TICKERS = 3

VALID_INTENTS = {"stock_query", "open_ended", "unclear", "portfolio_query"}

# Iteration 4, Sezione 4, Problema 27: how many recent messages of
# conversation history to pass into the LLM calls below, so implicit
# follow-up references (e.g. "it", "its main rival") can be resolved
# against the previous turn instead of being classified as if the
# conversation had no prior context. Capped rather than unbounded so the
# prompt sent on every turn doesn't grow with the whole session — a small,
# fixed window is enough to resolve the immediate-previous-turn references
# actually seen in manual testing (Nvidia -> "its main rival in GPUs"),
# and keeps the added prompt length/cost bounded and predictable.
MAX_HISTORY_MESSAGES = 6


def _history_messages(history: list[dict] | None) -> list[dict]:
    """
    Normalize a conversation history (as stored in st.session_state.messages
    — a list of {"role": "user"/"assistant", "content": str} dicts) into the
    last MAX_HISTORY_MESSAGES entries, in OpenAI chat message format.

    Returns [] for None/empty input, so every caller below that accepts an
    optional history parameter behaves exactly as it did before this
    parameter existed when no history is passed — this is what keeps all
    pre-Problema-27 callers and tests unaffected.
    """
    if not history:
        return []
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    return [{"role": m["role"], "content": m["content"]} for m in trimmed]

COMMON_TICKER_FIXES = {
    "FORD": "F",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "FACEBOOK": "META",
    "GENERALMOTORS": "GM",
    "BERKSHIRE": "BRK.B",
    "BERKSHIREHATHAWAY": "BRK.B",
}


def classify_query_intent(query: str, history: list[dict] | None = None) -> str:
    """
    Classify the user's query into one of four intents:

      - "stock_query":     the query names or clearly implies specific
                         company/companies (e.g. "What is Tesla's P/E?",
                         "Compare AAPL and MSFT", or a product/brand name
                         that unambiguously points to one company, e.g.
                         "Should I buy an iPhone maker?" -> Apple).
      - "open_ended":    the query asks for general investment advice
                         without naming or implying a specific stock (e.g.
                         "What should I invest in?", "Is now a good time to
                         buy stocks?").
      - "portfolio_query": the query asks about the user's OWN tracked
                         holdings as a whole, without naming a specific
                         ticker to look up (e.g. "How is my portfolio
                         doing?", "What's my P&L?", "Should I rebalance?").
                         Naming a specific ticker together with "my" (e.g.
                         "How is my AAPL holding doing?") still counts as
                         stock_query — this label is only for questions
                         about the portfolio as a whole.
      - "unclear":       the query is off-topic, empty of financial meaning,
                         or too ambiguous to act on.

     Note: this prompt must recognize product/brand
    references the same way the ticker extractor's prompt does (see
    _extract_all_tickers below). An earlier version only mentioned "clearly
    implies" without an example, and in practice the model classified
    product-reference queries like "Should I buy an iPhone maker?" as
    open_ended instead of stock_query, which routed them to a generic
    clarification message and never gave the ticker extractor a chance to run at all. The explicit example below keeps the two prompts' behavior consistent.

    Iteration 4 Sezione 4 addition (Problema 26): before this label existed,
    any portfolio-level question with no named ticker fell through to
    "unclear" and got the generic clarification fallback, even though
    app.py already has all the data needed to answer it via
    compute_portfolio_summary() — the intent classifier (Iteration 1) and
    the Portfolio Tracker (Iteration 3) had simply never been connected.

    Defaults to "stock_query" on classification failure, so a downstream
    ticker-extraction miss (rather than a silent misclassification) is what
    surfaces to the user — this keeps failures visible instead of masking
    them behind a generic clarification message.

    Iteration 4 Sezione 4 addition (Problema 27): accepts an optional
    `history` (recent conversation turns, see _history_messages above), so
    a follow-up with no explicit company reference of its own can still be
    classified correctly by looking at what was just discussed, instead of
    being judged in isolation. Defaults to None so every pre-Problema-27
    caller/test is unaffected.
    """
    try:
        history_messages = _history_messages(history)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=5,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's financial query into exactly one label: "
                        "stock_query, open_ended, portfolio_query, or unclear.\n"
                        "- stock_query: names or clearly implies one or more specific "
                        "companies/tickers. This includes an unambiguous product or "
                        "brand reference that points to one company, even if the "
                        "company itself is never named (e.g. 'Should I buy an iPhone "
                        "maker?' implies Apple; 'Is the Windows maker a good buy?' "
                        "implies Microsoft) — classify these as stock_query, not "
                        "open_ended. It also includes a follow-up that only makes sense "
                        "in light of the conversation shown before the final message "
                        "below (e.g. 'How does it compare to its main rival?' right "
                        "after discussing a specific company) — classify these as "
                        "stock_query too, not unclear.\n"
                        "- open_ended: asks for general investing advice with no "
                        "specific company named OR implied, even after considering any "
                        "conversation shown below.\n"
                        "- portfolio_query: asks about the user's OWN tracked "
                        "portfolio/holdings as a whole, with no specific ticker named "
                        "(e.g. 'How is my portfolio doing?', 'What's my total P&L?', "
                        "'Should I rebalance?'). If a specific ticker IS named "
                        "alongside 'my' (e.g. 'How is my AAPL holding doing?'), use "
                        "stock_query instead.\n"
                        "- unclear: off-topic, empty, or too ambiguous to act on even "
                        "with the conversation shown below.\n"
                        "If prior conversation turns are shown before the final "
                        "message, they are only for resolving references in that final "
                        "message — classify the final message only.\n"
                        "Reply with ONLY the label, nothing else."
                    ),
                },
                *history_messages,
                {"role": "user", "content": query},
            ],
        )
        label = response.choices[0].message.content.strip().lower()
        return label if label in VALID_INTENTS else "stock_query"
    except Exception:
        return "stock_query"


def extract_ticker_from_query(query: str) -> str | None:
    """
    Backward-compatible single-ticker extractor, kept for callers that only
    need one symbol. Internally delegates to extract_tickers_from_query and
    returns the first match.
    """
    tickers = extract_tickers_from_query(query)
    return tickers[0] if tickers else None


def _extract_all_tickers(query: str, history: list[dict] | None = None) -> list[str]:
    """
    Internal helper: makes the single LLM call used by ticker extraction and
    returns the FULL de-duplicated, corrected list of tickers found — before
    the MAX_TICKERS cap is applied. Both extract_tickers_from_query() and
    extract_tickers_with_truncation_info() build on this so the LLM is only
    called once per query regardless of which public function is used.

    Iteration 4 Sezione 4 addition (Problema 27): accepts an optional
    `history` (recent conversation turns) so a query with no company of its
    own — e.g. "How does it compare to its main rival in GPUs?" right after
    a Nvidia query — can resolve "it" against the company just discussed,
    instead of always returning [] for a query with no ticker/name of its
    own. This is the extraction half of the same fix as the history support
    added to classify_query_intent() above; both were needed, since a query
    like this was previously misrouted twice over — first by isolated
    intent classification, then again by isolated ticker extraction. This
    was the more consequential of the two: even once intent classification
    is fixed, extraction still needs the same context to name a ticker at
    all. Defaults to None so every pre-Problema-27 caller/test (which never
    passed a history argument) is unaffected.

    Only companies the user actually named (or unambiguously referenced,
    e.g. by product name) are extracted — the model is explicitly told not
    to add extra competitors or "for comparison" companies that were never
    mentioned, since that produced unrequested results such as adding GM to
    a "Compare Tesla and Ford" query.

    IMPORTANT: the extraction prompt below must NOT itself cap the result at
    MAX_TICKERS. An earlier version told the model to extract "up to a
    maximum of 3", which made the model self-truncate during extraction —
    so this function never actually returned more than 3 tickers, even when
    the user named 4+ companies. That silently broke the truncation
    notice: extract_tickers_with_truncation_info() detects truncation by
    checking len(all_tickers) > MAX_TICKERS on THIS function's output, so if
    the model already capped it at 3, that check can never fire and the
    user is never told a company was dropped (see Diario Tecnico). The cap
    must be applied once, downstream, by the callers below — never here.
    """
    try:
        history_messages = _history_messages(history)
        history_note = (
            "If prior conversation turns are shown before the final message, use "
            "them ONLY to resolve pronouns or implicit references in the final "
            "message (e.g. 'it', 'its main rival', 'the same company') to a "
            "company actually named earlier — then extract that company's "
            "ticker. Never pull in an extra company from the conversation history "
            "that the final message does not itself refer to, implicitly or "
            "explicitly. "
            if history_messages else ""
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=40,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial ticker extractor. "
                        "Given a user query, identify stock ticker symbols ONLY for "
                        "companies explicitly named or unambiguously referenced in the "
                        "query itself (e.g. a product name like 'iPhone' clearly "
                        "implies Apple). Do NOT add competitors, related companies, or "
                        "any other company for context or comparison purposes — extract "
                        "every company the user actually mentioned, with NO upper limit "
                        "on how many you return (a separate step outside your control "
                        "handles any limit on how many are compared at once, and needs "
                        "to know the true full count, so do not cap or truncate your "
                        "answer yourself). "
                        f"{history_note}"
                        "If you find no companies at all, return NONE — never pad the "
                        "list with a placeholder. "
                        "Always use the REAL stock exchange ticker symbol, never the "
                        "company name written in capital letters. For example: Ford "
                        "Motor Company's ticker is F, not FORD; Alphabet/Google's "
                        "ticker is GOOGL, not GOOGLE or ALPHABET; Meta/Facebook's "
                        "ticker is META, not FACEBOOK. "
                        "Reply with ONLY a comma-separated list of uppercase ticker "
                        "symbols (e.g. 'AAPL,MSFT,GOOGL'), with no spaces and no "
                        "other text. The word NONE must appear only as the entire "
                        "reply on its own, never mixed in with real tickers, and only "
                        "when no ticker at all can be identified."
                    ),
                },
                *history_messages,
                {"role": "user", "content": query},
            ],
        )
        result = response.choices[0].message.content.strip().upper()
        if result == "NONE" or not result:
            return []
        tickers = [t.strip() for t in result.split(",") if t.strip()]
        tickers = [t for t in tickers if t[:1].isalpha() and "NONE" not in t]
        tickers = [COMMON_TICKER_FIXES.get(t, t) for t in tickers]
        seen = set()
        deduped = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped
    except Exception:
        return []


def extract_tickers_from_query(query: str, history: list[dict] | None = None) -> list[str]:
    """
    Use a zero-temperature LLM call to extract up to MAX_TICKERS stock
    tickers from the user's natural language query. Returns a list of
    uppercase ticker strings (e.g. ['AAPL', 'MSFT']), or [] if none found.

    Kept as a simple, backward-compatible entry point. Callers that need to
    know whether the user actually mentioned more companies than the app
    supports (to surface that to the user, rather than silently dropping
    them) should use extract_tickers_with_truncation_info() instead.

    `history` (Iteration 4 Sezione 4, Problema 27): optional recent
    conversation turns, forwarded to _extract_all_tickers() to resolve
    implicit follow-up references. Defaults to None, unchanged behavior.
    """
    return _extract_all_tickers(query, history)[:MAX_TICKERS]


def extract_tickers_with_truncation_info(
    query: str, history: list[dict] | None = None
) -> tuple[list[str], bool]:
    """
    Same extraction as extract_tickers_from_query(), but also reports
    whether the query mentioned more companies than MAX_TICKERS supports.

    Returns (tickers, was_truncated) where tickers is capped at MAX_TICKERS
    and was_truncated is True if additional companies had to be dropped.
    This lets the UI tell the user "only comparing the first 3" instead of
    silently discarding a company — which previously led the LLM to
    fabricate a misleading explanation (e.g. claiming a company's data was
    unavailable when it was simply never requested).

    `history` (Iteration 4 Sezione 4, Problema 27): optional recent
    conversation turns, forwarded to _extract_all_tickers() to resolve
    implicit follow-up references. Defaults to None, unchanged behavior.
    """
    all_tickers = _extract_all_tickers(query, history)
    return all_tickers[:MAX_TICKERS], len(all_tickers) > MAX_TICKERS



# Iteration 3, inclusive design improvement: always instruct the model to answer in the 
# language the question was asked in.
LANGUAGE_MATCH_INSTRUCTION = (
    "Always answer in the same language the user's question was written in. "
    "If the question mixes languages or the language is ambiguous, default to English.\n"
)

# Iteration 3, inclusive design improvement: an optional simplified-language
# mode, toggled by the user in the UI.
SIMPLIFIED_MODE_INSTRUCTION = (
    "The user has requested simplified explanations: avoid financial jargon "
    "where possible, and whenever a technical term is genuinely unavoidable "
    "(e.g. 'P/E ratio'), briefly define it in plain language the first time "
    "it is used. Prefer short sentences.\n"
)


def build_prompt(
    stock_data,
    user_query: str,
    news_context: str = "",
    simplified_mode: bool = False,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Construct the message list for the OpenAI Chat API.

    Accepts either a single stock_data dict (single-ticker path, kept for
    backward compatibility) or a list of stock_data dicts (multi-ticker
    comparative path), and builds the appropriate context block.

    Iteration 2: an optional news_context string (built by
    news_data.build_news_context) can be appended after the financial data
    block, so NewsAPI headlines are available to the model as grounding
    context alongside the yfinance-derived figures. Defaults to "" so
    existing callers (and Iteration 1 tests) that don't pass it are
    unaffected.

    Iteration 3: language-matching is always applied; simplified_mode is an
    opt-in flag (default False, so existing callers and tests are
    unaffected) that adds the plain-language instruction above.

    Iteration 4 Sezione 4 addition (Problema 27): an optional `history`
    (recent conversation turns) is inserted between the system prompt and
    the current user question, so the final answer itself can also be
    phrased with awareness of what was just discussed (e.g. an explicit
    "compared to Nvidia, which you just asked about" instead of reading as
    a reply with no memory of the conversation) — completing the same fix
    already applied to intent classification and ticker extraction above.
    Defaults to None, so every pre-Problema-27 caller/test producing a
    two-message [system, user] list is unaffected.
    """
    if isinstance(stock_data, list):
        data_context = build_comparative_context(stock_data)
        present_names = ", ".join(f"{d.get('ticker')} ({d.get('name', '')})" for d in stock_data)
        instruction = (
            "CRITICAL RULE, follow this before anything else below: the user's question "
            "may name more companies than are present in the DATA block below (the app "
            "only supports comparing a limited number at a time, and the UI already tells "
            "the user this separately). Your answer must ONLY discuss the companies that "
            "are actually present in the DATA block — do not name, mention, or reference "
            "any other company from the question in any way, not even to note it is "
            f"missing, unavailable, or excluded. The companies present in the data are: "
            f"{present_names}. Treat the question as if it had only asked about these.\n\n"
            "You are a financial advisor assistant. Your role is to help non-technical "
            "retail investors understand and compare stocks. "
            "You always base your analysis strictly on the retrieved financial data provided "
            "in the context block below — never invent numbers or cite data not present in the context. "
            "Explicitly compare the companies across the metrics given "
            "(valuation, growth, risk) rather than describing each one in isolation. "
            "Explain your reasoning in plain language. Always include a brief risk disclaimer. "
            "Keep responses concise and structured.\n\n"
        )
    else:
        data_context = build_data_context(stock_data)
        instruction = (
            "You are a financial advisor assistant. Your role is to help non-technical "
            "retail investors understand stocks and make more informed decisions. "
            "You always base your analysis strictly on the retrieved financial data provided "
            "in the context block below — never invent numbers or cite data not present in the context. "
            "Explain your reasoning in plain language. Always include a brief risk disclaimer. "
            "Keep responses concise and structured.\n\n"
        )

    accessibility_instructions = LANGUAGE_MATCH_INSTRUCTION
    if simplified_mode:
        accessibility_instructions += SIMPLIFIED_MODE_INSTRUCTION

    system_prompt = f"{instruction}{accessibility_instructions}\n{data_context}{news_context}"

    return [
        {"role": "system", "content": system_prompt},
        *_history_messages(history),
        {"role": "user", "content": user_query},
    ]
