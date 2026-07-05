"""
rag_pipeline.py
RAG Pipeline layer — ticker extraction and prompt construction.
"""

import os
from openai import OpenAI
from src.financial_data import build_data_context

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_ticker_from_query(query: str) -> str | None:
    """
    Use a zero-temperature LLM call to extract a stock ticker from the user's
    natural language query. Returns the ticker string (e.g. 'AAPL') or None.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=10,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial ticker extractor. "
                        "Given a user query about a stock or company, "
                        "reply with ONLY the ticker symbol in uppercase (e.g. AAPL, TSLA, MSFT). "
                        "If no ticker can be identified, reply with exactly: NONE"
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        result = response.choices[0].message.content.strip().upper()
        return None if result == "NONE" else result
    except Exception:
        return None


def build_prompt(stock_data: dict, user_query: str) -> list:
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
