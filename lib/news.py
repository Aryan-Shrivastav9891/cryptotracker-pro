"""Crypto news (CryptoCompare) + optional VADER headline sentiment.

No API key required. Sentiment is used only as a SOFT, clearly-labeled tilt on the
forecast signal — never the main driver. If vaderSentiment isn't installed, the
sentiment functions degrade gracefully to "Neutral / unavailable".
"""
from __future__ import annotations

from typing import Any, Dict, List

import requests
import streamlit as st

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _ANALYZER = SentimentIntensityAnalyzer()
    HAS_VADER = True
except Exception:  # pragma: no cover
    _ANALYZER = None
    HAS_VADER = False

NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
_HEADERS = {"Accept": "application/json", "User-Agent": "CryptoTrackerPro/1.0 (Streamlit)"}
_TIMEOUT = 20
_POS, _NEG = 0.05, -0.05  # VADER's conventional thresholds


@st.cache_data(ttl=600, show_spinner=False)
def get_news(symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Latest news articles for a coin symbol. Cached for 10 minutes.

    Returns a list of dicts with keys: title, body, url, imageurl, published_on,
    source. Never raises — returns [] on any failure.
    """
    if not symbol:
        return []
    try:
        resp = requests.get(
            NEWS_URL,
            params={"categories": symbol.upper()},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("Data", []) or []
        return data[:limit]
    except requests.RequestException:
        return []


def analyze_sentiment(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate VADER sentiment over headlines (+ a little body text).

    Returns {score, label, bullish, neutral, bearish, available}. ``score`` is the
    mean VADER compound in [-1, 1]; ``label`` is Bullish / Neutral / Bearish.
    Falls back to a neutral, ``available=False`` result if VADER is missing or
    there are no articles.
    """
    neutral = {"score": 0.0, "label": "Neutral", "bullish": 0, "neutral": 0,
               "bearish": 0, "available": False}
    if not articles or not HAS_VADER:
        return neutral

    scores: List[float] = []
    bull = neu = bear = 0
    for art in articles:
        title = art.get("title") or ""
        body = (art.get("body") or "")[:300]
        text = (title + ". " + body).strip()
        if not text:
            continue
        compound = _ANALYZER.polarity_scores(text)["compound"]
        scores.append(compound)
        if compound > _POS:
            bull += 1
        elif compound < _NEG:
            bear += 1
        else:
            neu += 1

    if not scores:
        return neutral

    avg = sum(scores) / len(scores)
    label = "Bullish" if avg > _POS else "Bearish" if avg < _NEG else "Neutral"
    return {"score": avg, "label": label, "bullish": bull, "neutral": neu,
            "bearish": bear, "available": True}
