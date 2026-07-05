# Financial Advisor Bot

**University of London — BSc Computer Science — CM3070 Final Project**
Template 4.2 — Financial Advisor Bot (CM3020 Artificial Intelligence)

---

## Overview

A conversational AI assistant that helps retail investors understand financial data and make more informed investment decisions. The system uses a **Retrieval-Augmented Generation (RAG)** architecture: it retrieves live market data for any publicly traded stock and uses a Large Language Model (GPT-4o-mini) to reason over that data and generate plain-language analysis.

The interface is a browser-based chat application built with Streamlit.

---

## Project Status

This repository tracks the incremental development of the system across Phase 2 of the project (July–September 2026).

| Iteration | Description | Status |
|---|---|---|
| 0 — Baseline | Simple prototype from preliminary report | ✅ Current |
| 1 — Multi-ticker + Intent detection | Handle multi-stock queries and open-ended questions | ⬜ Planned |
| 2 — NewsAPI + Visualisation | Real-time news and interactive price charts | ⬜ Planned |
| 3 — Memory + Portfolio tracker | SQLite persistence and portfolio P&L | ⬜ Planned |
| 4 — Sentiment + Backtesting | VADER sentiment analysis and historical backtesting | ⬜ Planned |
| 5 — User testing | 5-participant study, 20-query evaluation | ⬜ Planned |
| 6 — Final polish | Documentation, refactoring, submission prep | ⬜ Planned |

---

## Architecture

```
User query
    │
    ▼
Ticker Extraction (LLM, temperature=0)
    │
    ▼
Data Retrieval (yfinance → Yahoo Finance)
    │
    ▼
RAG Prompt Construction (context + query)
    │
    ▼
LLM Reasoning (GPT-4o-mini)
    │
    ▼
Response + Disclaimer (Streamlit UI)
```

### File structure

```
financial-advisor-bot/
├── app.py                  # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── financial_data.py   # Data retrieval layer (yfinance)
│   ├── rag_pipeline.py     # RAG pipeline: ticker extraction + prompt construction
│   └── advisor.py          # LLM reasoning layer (OpenAI API)
├── tests/                  # pytest unit and integration tests (added in Iteration 1)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── .gitignore
```

---

## Setup

### Requirements
- Python 3.10 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/financial-advisor-bot.git
cd financial-advisor-bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# Then edit .env and replace sk-your-api-key-here with your actual key
# On Mac/Linux you can also use:
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 4. Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Usage

Type any question about a publicly traded stock in the chat input. Examples:

- *"Should I buy Apple stock?"*
- *"What is Tesla's P/E ratio?"*
- *"Tell me about NVDA"*
- *"What do analysts think about Microsoft?"*

The bot will retrieve live data from Yahoo Finance and provide an analysis grounded in that data.

---

## Disclaimer

⚠️ This tool is for **educational purposes only** and does not constitute regulated financial advice. Always consult a qualified financial advisor before making investment decisions. Past performance is not a reliable indicator of future results.

---

## Academic context

This project is submitted as part of the CM3070 Final Project module, BSc Computer Science, University of London. The RAG-based approach was chosen over classical reinforcement learning for financial advisory, as it provides more reliable, data-grounded responses with lower infrastructure requirements — a design decision documented and justified in the project report.
