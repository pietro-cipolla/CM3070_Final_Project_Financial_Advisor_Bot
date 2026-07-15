"""
rag_pipeline.py
RAG Pipeline layer — query intent classification, ticker extraction and
prompt construction.

Iteration 1 additions:
  1. Multi-ticker extraction: a query can now reference up to 3 companies
     (e.g. "Compare Apple, Microsoft and Google"), instead of only the first
     ticker found.
  2. Query intent classification: queries are classified before ticker
     extraction runs, so open-ended / off-topic queries can be routed to a
     clarification prompt instead of silently failing or hallucinating
     an answer with no financial grounding.
"""

import os
from openai import OpenAI
from src.financial_data import build_data_context, build_comparative_context

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TICKERS = 3

VALID_INTENTS = {"stock_query", "open_ended", "unclear"}


def classify_query_intent(query: str) -> str:
    """
    Classify the user's query into one of three intents:

      - "stock_query":  the query names or clearly implies specific
                         company/companies (e.g. "What is Tesla's P/E?",
                         "Compare AAPL and MSFT").
      - "open_ended":    the query asks for general investment advice
                         without naming a specific stock (e.g. "What should
                         I invest in?", "Is now a good time to buy stocks?").
      - "unclear":       the query is off-topic, empty of financial meaning,
                         or too ambiguous to act on.

    Defaults to "stock_query" on classification failure, so a downstream
    ticker-extraction miss (rather than a silent misclassification) is what
    surfaces to the user — this keeps failures visible instead of masking
    them behind a generic clarification message.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=5,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's financial query into exactly one label: "
                        "stock_query, open_ended, or unclear.\n"
                        "- stock_query: names or clearly implies one or more specific companies/tickers.\n"
                        "- open_ended: asks for general investing advice with no specific company named.\n"
                        "- unclear: off-topic, empty, or too ambiguous to act on.\n"
                        "Reply with ONLY the label, nothing else."
                    ),
                },
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


def extract_tickers_from_query(query: str) -> list[str]:
    """
    Use a zero-temperature LLM call to extract up to MAX_TICKERS stock
    tickers from the user's natural language query. Returns a list of
    uppercase ticker strings (e.g. ['AAPL', 'MSFT']), or [] if none found.

    Only companies the user actually named (or unambiguously referenced,
    e.g. by product name) are extracted — the model is explicitly told not
    to add extra competitors or "for comparison" companies that were never
    mentioned. Bug found in manual testing: "Compare Tesla and Ford" was
    returning TSLA, F, AND GM (an unrequested competitor) before this fix.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=20,
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
                        "only what the user actually mentioned, up to a maximum of 3. "
                        "If you find fewer than 3 companies, return only the ones you "
                        "found — never pad the list with a placeholder. "
                        "Reply with ONLY a comma-separated list of uppercase ticker "
                        "symbols (e.g. 'AAPL,MSFT,GOOGL'), with no spaces and no "
                        "other text. The word NONE must appear only as the entire "
                        "reply on its own, never mixed in with real tickers, and only "
                        "when no ticker at all can be identified."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        result = response.choices[0].message.content.strip().upper()
        if result == "NONE" or not result:
            return []
        tickers = [t.strip() for t in result.split(",") if t.strip()]
        # Defensive filter: even with the prompt above, the model has been
        # observed padding a short list with a stray "NONE" / ".NONE" token
        # instead of just returning the real tickers it found. Drop anything
        # that isn't a plausible ticker (must start with a letter and must
        # not contain "NONE") rather than trusting the model's formatting.
        tickers = [t for t in tickers if t[:1].isalpha() and "NONE" not in t]
        # De-duplicate while preserving order, cap at MAX_TICKERS
        seen = set()
        deduped = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped[:MAX_TICKERS]
    except Exception:
        return []


def build_prompt(stock_data, user_query: str) -> list[dict]:
    """
    Construct the message list for the OpenAI Chat API.

    Accepts either a single stock_data dict (single-ticker path, kept for
    backward compatibility) or a list of stock_data dicts (multi-ticker
    comparative path), and builds the appropriate context block.
    """
    if isinstance(stock_data, list):
        data_context = build_comparative_context(stock_data)
        instruction = (
            "You are a financial advisor assistant. Your role is to help non-technical "
            "retail investors understand and compare stocks. "
            "You always base your analysis strictly on the retrieved financial data provided "
            "in the context block below — never invent numbers or cite data not present in the context. "
            "When multiple companies are present, explicitly compare them across the metrics given "
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

    system_prompt = f"{instruction}{data_context}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
