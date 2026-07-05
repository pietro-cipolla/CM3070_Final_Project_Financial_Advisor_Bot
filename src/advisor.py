"""
advisor.py
LLM Reasoning layer — sends the RAG prompt to OpenAI and returns the response.
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_advice(messages: list) -> str:
    """
    Call the OpenAI Chat Completions API with the constructed RAG prompt.
    Returns the assistant's response as a string, or a user-friendly error message.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=600,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        err = str(e).lower()
        if "auth" in err or "api key" in err:
            return "⚠️ Authentication error: please check your OPENAI_API_KEY in the .env file."
        if "rate" in err:
            return "⚠️ Rate limit reached. Please wait a moment and try again."
        return f"⚠️ An error occurred while contacting the language model: {e}"
