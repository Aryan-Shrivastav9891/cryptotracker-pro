"""Crypto news with a resilient 3-tier fallback + VADER sentiment.

1. PRIMARY  — CryptoCompare / CoinDesk Data API "latest articles" (needs a free
   key; read from env CRYPTOCOMPARE_API_KEY / COINDESK_DATA_API_KEY or st.secrets;
   sent both as ?api_key= and as `Authorization: Apikey <KEY>`).
2. FALLBACK — keyless RSS via feedparser (CoinDesk, Cointelegraph, Decrypt),
   filtered to headlines mentioning the coin's symbol/name. Works with NO key.
3. If everything fails — return an empty result; the UI shows "news unavailable".

Sentiment is scored per headline with vaderSentiment (graceful neutral fallback).
Never the main driver of any trading signal — only a small, labeled tilt.
"""
from __future__ import annotations

import calendar
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _ANALYZER = SentimentIntensityAnalyzer()
    HAS_VADER = True
except Exception:  # pragma: no cover
    _ANALYZER = None
    HAS_VADER = False

try:
    import feedparser  # type: ignore

    HAS_FEEDPARSER = True
except Exception:  # pragma: no cover
    feedparser = None  # type: ignore
    HAS_FEEDPARSER = False

_HEADERS = {"Accept": "application/json", "User-Agent": "CryptoTrackerPro/1.0"}
_TIMEOUT = 20
_POS, _NEG = 0.05, -0.05
_CC_URL = "https://min-api.cryptocompare.com/data/v2/news/"
_RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}


def _api_key() -> Optional[str]:
    for env in ("CRYPTOCOMPARE_API_KEY", "COINDESK_DATA_API_KEY"):
        if os.environ.get(env):
            return os.environ[env]
    try:  # st.secrets may not exist / may raise if no secrets file
        for k in ("CRYPTOCOMPARE_API_KEY", "COINDESK_DATA_API_KEY"):
            if k in st.secrets:
                return st.secrets[k]
    except Exception:
        pass
    return None


def _score(text: str) -> float:
    if not text or not HAS_VADER:
        return 0.0
    return float(_ANALYZER.polarity_scores(text)["compound"])


def _label(score: float) -> str:
    return "Bullish" if score > _POS else "Bearish" if score < _NEG else "Neutral"


def _normalize(title, url, source, ts, body) -> Dict[str, Any]:
    snippet = (body or "")[:200].strip()
    when = ""
    if ts:
        try:
            when = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            when = ""
    title = title or "Untitled"
    return {
        "title": title,
        "url": url or "#",
        "source": source or "",
        "published": when,
        "body_snippet": snippet,
        "sentiment": _score(f"{title}. {snippet}"),
    }


# --------------------------------------------------------------------------- #
# Tier 1 — CryptoCompare / CoinDesk Data API (with key)
# --------------------------------------------------------------------------- #
def _fetch_cryptocompare(symbol: str, limit: int) -> Optional[List[Dict[str, Any]]]:
    key = _api_key()
    if not key:
        return None
    try:
        resp = requests.get(
            _CC_URL,
            params={"categories": symbol.upper(), "api_key": key},
            headers={**_HEADERS, "Authorization": f"Apikey {key}"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("Data", []) or []
        out = [_normalize(a.get("title"), a.get("url"), a.get("source_info", {}).get("name")
                          or a.get("source"), a.get("published_on"), a.get("body")) for a in data]
        return out[:limit] if out else None
    except requests.RequestException:
        return None


# --------------------------------------------------------------------------- #
# Tier 2 — keyless RSS
# --------------------------------------------------------------------------- #
def _fetch_rss(symbol: str, name: Optional[str], limit: int) -> List[Dict[str, Any]]:
    if not HAS_FEEDPARSER:
        return []
    terms = {symbol.lower()}
    if name:
        terms.add(name.lower())
    out: List[Dict[str, Any]] = []
    for source, url in _RSS_FEEDS.items():
        try:
            # Fetch via requests for a hard timeout, then parse the bytes.
            resp = requests.get(url, headers={"User-Agent": _HEADERS["User-Agent"]}, timeout=_TIMEOUT)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception:
            continue
        for entry in getattr(feed, "entries", [])[:40]:
            title = entry.get("title", "") or ""
            summary = entry.get("summary", "") or ""
            hay = (title + " " + summary).lower()
            # Word-boundary match so short tickers ("AMP") don't hit "Sample"/"Ampere".
            if not any(re.search(r"\b" + re.escape(term) + r"\b", hay) for term in terms):
                continue
            ts = None
            if entry.get("published_parsed"):
                try:
                    # published_parsed is UTC; timegm reads it as UTC (mktime would assume local).
                    ts = calendar.timegm(entry["published_parsed"])
                except Exception:
                    ts = None
            out.append(_normalize(title, entry.get("link"), source, ts, summary))
    # newest first when we have dates
    out.sort(key=lambda a: a["published"], reverse=True)
    return out[:limit]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=1800, show_spinner=False)
def get_articles(symbol: str, name: Optional[str] = None, limit: int = 8) -> List[Dict[str, Any]]:
    """Return up to ``limit`` news articles (with per-article sentiment).

    Tries the keyed API first, then keyless RSS, then []. Cached for 30 minutes.
    """
    if not symbol:
        return []
    primary = _fetch_cryptocompare(symbol, limit)
    if primary:
        return primary
    return _fetch_rss(symbol, name, limit)


@st.cache_data(ttl=1800, show_spinner=False)
def news_sentiment(symbol: str, name: Optional[str] = None, limit: int = 8) -> Dict[str, Any]:
    """Aggregate sentiment over the latest articles.

    Returns {score, label, n, items, available}. ``score`` is the mean VADER
    compound in [-1, 1]; ``label`` is Bullish / Neutral / Bearish.
    """
    items = get_articles(symbol, name=name, limit=limit)
    if not items:
        return {"score": 0.0, "label": "Neutral", "n": 0, "items": [], "available": False}
    scores = [a["sentiment"] for a in items]
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "score": avg,
        "label": _label(avg),
        "n": len(items),
        "items": items,
        "available": True,
    }
