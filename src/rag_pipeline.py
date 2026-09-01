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
import re
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

# Problema 37 (Iteration 4, Sezione 4): last-resort pattern used ONLY when
# the LLM extraction call itself returned no ticker at all (see
# _naive_ticker_candidates below). Matches a token already written in the
# exact "SYMBOL.SUFFIX" shape of a real Yahoo Finance non-US ticker (e.g.
# "PST.MI", "ISP.MI", "BMW.DE") — i.e. the user already typed a ready-made,
# correctly-formed ticker, and the LLM extractor still failed to recognize
# it as one (confirmed case: "PST.MI" for Poste Italiane, not among the
# extraction prompt's worked examples).
#
# Deliberately narrow: requires the literal dot + 1-3 letter exchange
# suffix already present in the text, rather than matching any bare
# uppercase word (which would also match ordinary acronyms like "CEO" or
# "P/E" and risk resolving a false positive to some unrelated real
# company via resolve_ticker()'s yfinance.Search() call). A bare US
# ticker with no dot (e.g. a query that is just "AAPL") is NOT matched
# here — that case is already handled correctly by the LLM extractor in
# manual testing, so this fallback is scoped to the specific gap that was
# actually demonstrated, not a general-purpose ticker detector.
#
# The symbol part allows digits as well as letters (Iteration 4 Sezione 4,
# testing after Problema 38: several Asian exchanges — confirmed on Tokyo,
# e.g. Toyota's "7203.T" — use a purely numeric local code instead of a
# letter-based one; the original letters-only class would not match a
# user-typed "7203.T" any more than it would recognize it elsewhere, see
# the isalnum() fix in _extract_all_tickers_with_names below). The suffix
# itself stays letters-only, since every real exchange suffix is (.T,
# .MI, .DE, ...) — only ever the local instrument code itself is
# sometimes numeric.
_SUFFIXED_TICKER_RE = re.compile(r"\b[A-Z0-9]{1,6}\.[A-Z]{1,3}\b")


def _naive_ticker_candidates(query: str) -> list[str]:
    """
    Scan the raw query text for tokens that already look like a complete,
    correctly-suffixed non-US ticker (see _SUFFIXED_TICKER_RE above).

    This is only ever consulted by _extract_all_tickers() as a fallback
    for the specific case where the LLM extraction call found nothing —
    it does not run, and cannot override, the normal LLM-based extraction
    path. Returns [] if the query contains no such token, so a query with
    no exchange-suffixed symbol in it is completely unaffected (same
    behavior as before this fix).
    """
    return _SUFFIXED_TICKER_RE.findall(query.upper())


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


# Iteration 4 Sezione 4, Problema 38 continued: anchors one entry of a
# "TICKER:Company Name" extraction reply. Matches a short (1-10 char),
# ALL-CAPS/digit/dot/dash token immediately followed by ":", itself
# immediately preceded by either the start of the string or a comma. Used
# by _extract_all_tickers_with_names() to split the reply into entries
# WITHOUT a naive comma-split, which breaks as soon as a company's own
# legal name contains a comma (e.g. "Block, Inc.") — see that function's
# comments for the confirmed failure this fixes.
_PAIR_START_RE = re.compile(r"(?:^|,)\s*([A-Z0-9.\-]{1,10}):")


def extract_ticker_from_query(query: str) -> str | None:
    """
    Backward-compatible single-ticker extractor, kept for callers that only
    need one symbol. Internally delegates to extract_tickers_from_query and
    returns the first match.
    """
    tickers = extract_tickers_from_query(query)
    return tickers[0] if tickers else None


def _extract_all_tickers_with_names(
    query: str, history: list[dict] | None = None
) -> list[dict]:
    """
    Iteration 4 Sezione 4 addition (Problema 38): the actual LLM-calling
    implementation behind ticker extraction. Returns the FULL
    de-duplicated, corrected list of {"ticker": str, "name": str | None}
    pairs found — before the MAX_TICKERS cap is applied — instead of bare
    ticker strings.

    Why a name is carried alongside each ticker: Problema 37 (below) fixed
    the case where the LLM found NO ticker at all for an already
    correctly-typed symbol. Problema 38 is the opposite shape of failure:
    the LLM DOES return a ticker, confidently, but the wrong one — found on
    "can you compare Poste Italiane and Nvidia?", which extracted "PT.MI"
    instead of "PST.MI" (correctly extracted moments earlier for the same
    company asked about alone — the model's guess for a non-US company not
    among the prompt's worked examples is not perfectly stable across
    phrasings). Problema 37's fix (financial_data.resolve_ticker()) could
    not recover this on its own: its retry searches Yahoo Finance using the
    wrong TICKER text itself ("PT.MI"), not the company name the user
    actually meant ("Poste Italiane") — searching for the wrong string
    predictably does not find the right company. Carrying the name through
    lets financial_data.get_stock_summary() retry with the actual company
    name when the ticker-based attempts still come up empty — see its
    `expected_name` parameter for exactly where and why this is scoped.

    _extract_all_tickers() (below) remains the stable, backward-compatible
    entry point returning plain ticker strings only — every existing
    caller/test that only ever needed ticker strings is unaffected; only
    this function and its new caller, extract_ticker_candidates(), carry
    the name through.
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
            max_tokens=80,
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
                        "For a company listed on a non-US exchange, you MUST include "
                        "the correct Yahoo Finance exchange suffix — a bare symbol "
                        "without it can resolve to a completely different, unrelated "
                        "company on Yahoo Finance. For example: Intesa Sanpaolo (Borsa "
                        "Italiana) is ISP.MI, not ISP; Pirelli (Borsa Italiana) is "
                        "PIRC.MI, not PIRC; common suffixes include .MI (Milan), .PA "
                        "(Paris), .DE (Xetra/Germany), .L (London), .AS (Amsterdam), "
                        ".SW (Switzerland), .HK (Hong Kong), .T (Tokyo). If you are not "
                        "confident of the correct exchange suffix for a non-US company, "
                        "still include your best-guess suffix rather than omitting it — "
                        "a wrong-but-present suffix is safer than a bare symbol, since a "
                        "bare non-US symbol risks silently resolving to an unrelated "
                        "company instead of failing visibly. "
                        "For EACH company, also give the company name you associate with "
                        "that ticker (in English, e.g. 'Apple Inc' or 'Poste Italiane'), "
                        "in the form TICKER:Company Name — this name is used downstream "
                        "to double-check your ticker guess against a live financial "
                        "database, so give your best understanding of the company even if "
                        "you are not fully confident the ticker itself is correct. Never "
                        "put a colon inside the company name itself. "
                        "Reply with ONLY a comma-separated list of TICKER:Company Name "
                        "pairs (e.g. 'AAPL:Apple Inc,MSFT:Microsoft Corporation'), with no "
                        "spaces around the commas and no other text. The word NONE must "
                        "appear only as the entire reply on its own, never mixed in with "
                        "real entries, and only when no company at all can be identified."
                    ),
                },
                *history_messages,
                {"role": "user", "content": query},
            ],
        )
        result = response.choices[0].message.content.strip()
        pairs = []
        if result and result.upper() != "NONE":
            seen = set()

            def _keep(ticker_part: str) -> bool:
                # Iteration 4 Sezione 4, Problema 38 continued: isalpha() ->
                # isalnum() so a purely numeric local ticker code (e.g.
                # Toyota's "7203.T" on the Tokyo exchange) is not silently
                # discarded — the original check was written to reject
                # stray artifacts like ".NONE" (Problema 11), which starts
                # with a non-alphanumeric character either way, so widening
                # it to isalnum() does not reopen that case.
                return bool(ticker_part) and ticker_part[:1].isalnum() and "NONE" not in ticker_part

            # Entries are anchored on the next short, ALL-CAPS "TICKER:"
            # token that immediately follows a comma (or the start of the
            # reply) — NOT on a naive split of the whole reply by comma.
            # This matters because a company's own legal name routinely
            # contains a comma before its own entity suffix (e.g. "Block,
            # Inc.", "Meta Platforms, Inc."); a plain comma-split would cut
            # such a name in half and misread its second half ("Inc.") as
            # an extra, fabricated ticker of its own -- the confirmed
            # failure on "what can you tell me about Block?", which produced tickers
            # ["SQ", "INC."] instead of one company. A company name can
            # never itself match this anchor: it is never written entirely
            # in capitals, and is never immediately followed by a colon
            # (the prompt above explicitly forbids a colon inside the name).
            entry_starts = list(_PAIR_START_RE.finditer(result))
            if entry_starts:
                for i, m in enumerate(entry_starts):
                    ticker_part = m.group(1).strip().upper()
                    value_start = m.end()
                    value_end = (
                        entry_starts[i + 1].start() if i + 1 < len(entry_starts) else len(result)
                    )
                    name_part = result[value_start:value_end].strip().rstrip(",").strip() or None
                    if not _keep(ticker_part):
                        continue
                    ticker_part = COMMON_TICKER_FIXES.get(ticker_part, ticker_part)
                    if ticker_part not in seen:
                        seen.add(ticker_part)
                        pairs.append({"ticker": ticker_part, "name": name_part})
            else:
                # Legacy bare-ticker format: no ":" anywhere in the reply
                # (what every pre-Problema-38 mocked test still simulates,
                # and what the model itself might still reply with despite
                # the updated prompt). Comma-splitting is safe here — with
                # no company name in the reply at all, there is nothing for
                # an embedded comma to corrupt.
                tokens = [t.strip() for t in result.split(",") if t.strip()]
                for tok in tokens:
                    ticker_part = tok.strip().upper()
                    if not _keep(ticker_part):
                        continue
                    ticker_part = COMMON_TICKER_FIXES.get(ticker_part, ticker_part)
                    if ticker_part not in seen:
                        seen.add(ticker_part)
                        pairs.append({"ticker": ticker_part, "name": None})
    except Exception:
        pairs = []

    if pairs:
        return pairs

    # Problema 37 (Iteration 4, Sezione 4): the LLM found nothing at all —
    # either it genuinely didn't recognize the query, or (as in the
    # confirmed "PST.MI" case) the API call itself failed/errored. Before
    # giving up and surfacing the "could not identify a stock ticker"
    # message to the user, check whether they already typed a complete,
    # correctly-suffixed ticker verbatim (see _naive_ticker_candidates) —
    # this is a purely local, no-network check, so it costs nothing when
    # it finds nothing either. No company name is known for a symbol
    # recovered this way (it came from a regex match on the raw query
    # text, not from the model's own understanding of the company), so
    # "name" is None here — financial_data.get_stock_summary() treats a
    # missing expected_name exactly like not being given one at all.
    naive = _naive_ticker_candidates(query)
    return [{"ticker": t, "name": None} for t in naive]


def _extract_all_tickers(query: str, history: list[dict] | None = None) -> list[str]:
    """
    Backward-compatible entry point: the same extraction as
    _extract_all_tickers_with_names() above, reduced to plain ticker
    strings only (order preserved, duplicates already removed by that
    function). Both extract_tickers_from_query() and
    extract_tickers_with_truncation_info() build on this so the LLM is only
    called once per query regardless of which public function is used.

    Iteration 4 Sezione 4 addition (Problema 27): accepts an optional
    `history` (recent conversation turns), forwarded unchanged to
    _extract_all_tickers_with_names(). See that function's docstring for
    the history-resolution behavior and the Problema 37/38 fallback chain
    — this wrapper only strips the name back off the result.
    """
    return [pair["ticker"] for pair in _extract_all_tickers_with_names(query, history)]


def extract_ticker_candidates(
    query: str, history: list[dict] | None = None
) -> tuple[list[dict], bool]:
    """
    Iteration 4 Sezione 4 addition (Problema 38): like
    extract_tickers_with_truncation_info() below, but each entry is a
    {"ticker": str, "name": str | None} pair instead of a bare ticker
    string, so app.py can forward the associated company name into
    financial_data.get_stock_summary()'s `expected_name` parameter (single-
    ticker path) or financial_data.get_multiple_stock_summaries()'s
    `expected_names` parameter (comparative path).

    Not specific to non-US tickers in how it works, even though the
    demonstrated failure (Problema 38: "PT.MI" guessed for Poste Italiane)
    was one — a wrong-and-nonexistent ticker guess for ANY market can only
    be recovered by searching for the company name instead of the (already
    shown to be wrong) symbol, so this plumbing is market-agnostic by
    construction, not scoped to a suffix check.

    Returns (pairs, was_truncated) with the same truncation semantics as
    extract_tickers_with_truncation_info(): pairs is capped at MAX_TICKERS,
    was_truncated is True if the query named more companies than that.
    """
    all_pairs = _extract_all_tickers_with_names(query, history)
    return all_pairs[:MAX_TICKERS], len(all_pairs) > MAX_TICKERS


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
