"""
rag_pipeline.py
RAG Pipeline layer — ticker extraction and prompt construction.

Iteration 1: extends single-ticker extraction to support up to 3 companies
in one query (e.g. "Compare Apple, Microsoft and Google"), instead of only
ever using the first ticker found.
"""

import os
from openai import OpenAI
from src.financial_data import build_data_context

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TICKERS = 3


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
                        "Given a user query, identify up to 3 stock ticker symbols "
                        "for the companies mentioned or clearly implied. "
                        "Reply with ONLY a comma-separated list of uppercase ticker "
                        "symbols (e.g. 'AAPL,MSFT,GOOGL'), with no spaces and no "
                        "other text. If no ticker can be identified, reply with "
                        "exactly: NONE"
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        result = response.choices[0].message.content.strip().upper()
        if result == "NONE" or not result:
            return []
        tickers = [t.strip() for t in result.split(",") if t.strip()]
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


def build_prompt(stock_data: dict, user_query: str) -> list[dict]:
    """
    Construct the message list for the OpenAI Chat API.
    Combines system role definition, retrieved financial context, and user query.
    """
    data_context = build_data_context(stock_data)

    system_prompt = (
        "You are a financial advisor assistant. Your role is to help non-technical "
        "retail investors understand stocks and make more informed decisions. "
        "You always base your analysis strictly on the retrieved financial data provided "
        "in the context block below — never invent numbers or cite data not present in the context. "
        "Explain your reasoning in plain language. Always include a brief risk disclaimer. "
        "Keep responses concise and structured.\n\n"
        f"{data_context}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
