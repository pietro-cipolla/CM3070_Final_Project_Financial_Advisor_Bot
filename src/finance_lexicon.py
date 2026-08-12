"""
finance_lexicon.py
A small, curated financial-domain lexicon override for VADER — Iteration 4.

VADER's sentiment lexicon is general-purpose, not finance-specific.
"""

FINANCIAL_LEXICON_OVERRIDE: dict[str, float] = {
    # Correction: VADER's general lexicon scores "beating" as violence
    # (-2.0). In financial headlines it is overwhelmingly "beating
    # expectations/estimates" — positive. Found in manual testing
    # (Ford, 2026-07-29): a headline about Ford's stock surging on an
    # earnings beat was scored negative because of this single word.
    "beating": 2.0,
    "beat": 1.5,
    "beats": 1.5,

    # Negative — analyst/rating actions and financial-distress language,
    # entirely absent from VADER's general lexicon. "downgrade" found
    # missing in manual testing (Salesforce/Levi Strauss, 2026-07-29).
    "downgrade": -1.8,
    "downgraded": -1.8,
    "downgrades": -1.8,
    "underperform": -1.2,
    "sell-off": -1.5,
    "selloff": -1.5,
    "plunge": -2.2,
    "plunged": -2.2,
    "plunges": -2.2,
    "tumble": -1.8,
    "tumbled": -1.8,
    "slump": -1.6,
    "slumped": -1.6,
    "miss": -1.0,
    "misses": -1.0,
    "missed": -1.0,
    "shortfall": -1.5,
    "writedown": -1.8,
    "write-down": -1.8,
    "layoffs": -1.8,
    "bankruptcy": -2.5,
    "default": -1.8,
    "recall": -1.3,
    "investigation": -1.2,
    "probe": -1.3,
    "antitrust": -1.0,
    "litigious": -1.0,
    "litigation": -1.2,
    "crash": -2.2,
    "crashed": -2.2,
    "overvalued": -1.0,
    "warns": -1.0,
    "warning": -1.0,
    "suspends": -1.2,
    "suspended": -1.2,
    "halted": -1.3,

    # Positive — analyst/rating actions and momentum language, likewise
    # mostly absent from VADER's general lexicon.
    "upgrade": 1.8,
    "upgraded": 1.8,
    "upgrades": 1.8,
    "outperform": 1.2,
    "rally": 1.5,
    "rallied": 1.5,
    "rallies": 1.5,
    "soar": 2.0,
    "soars": 2.0,
    "soared": 2.0,
    "surge": 1.8,
    "surges": 1.8,
    "surged": 1.8,
    "rebound": 1.3,
    "rebounded": 1.3,
    "bullish": 1.5,
    "buyback": 1.0,
    "overweight": 1.2,
    "resilient": 1.0,
}


def apply_financial_lexicon_override(analyzer) -> None:
    """
    Merge FINANCIAL_LEXICON_OVERRIDE into a VADER SentimentIntensityAnalyzer
    instance's lexicon IN PLACE, using the update mechanism VADER's own
    lexicon dict (a plain dict, confirmed via vaderSentiment's source)
    already supports natively — no monkeypatching, no forked dependency.

    Existing VADER words not in the override are left untouched; words
    present in both simply take the override's value — this is how
    "beating" is corrected without needing to patch VADER's own source
    lexicon file, and without affecting any other word VADER already
    scores reasonably.
    """
    analyzer.lexicon.update(FINANCIAL_LEXICON_OVERRIDE)
