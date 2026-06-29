"""Plain-English glossary of crypto / trading / stats jargon for hover tooltips.

Add a new term = ONE line in GLOSSARY. Lookups are case-insensitive and tolerate
plurals and extra spacing (see `normalize` / `get_definition`).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

GLOSSARY: Dict[str, str] = {
    "MAPE": "Average % the forecast was off by. Lower = better. 5% means predictions were about 5% away from the real price.",
    "MAE": "Average size of the error, in price units. Lower is better.",
    "RMSE": "Like MAE but punishes big misses harder. Lower is better.",
    "Directional accuracy": "How often the model got the up/down direction right. 50% = a coin toss.",
    "Skill score": "How much better the model is than a naive guess. 0 or below = no real edge.",
    "Backtest": "Testing the model on past real data it was NOT trained on.",
    "Walk-forward": "Backtesting step by step through time, always training only on the past.",
    "Confidence band": "The range the real price will likely stay inside. Wider = more uncertain.",
    "Volatility": "How wildly the price swings up and down.",
    "Random walk": "The idea that the next price move is basically unpredictable.",
    "Drift": "A slow steady tilt up or down on top of the random moves.",
    "Ensemble": "Combining several models so their average is steadier than any one alone.",
    "RSI": "Momentum meter from 0-100. Above 70 = possibly overbought, below 30 = possibly oversold.",
    "MACD": "A trend signal built from two moving averages crossing.",
    "Bollinger Bands": "Bands around the price showing how far it is stretched from its average.",
    "SMA": "Simple Moving Average - the plain average price over a chosen window.",
    "EMA": "Exponential Moving Average - like SMA but recent prices count more.",
    "Market cap": "Coin price multiplied by how many coins are in circulation.",
    "Circulating supply": "How many coins are actually available in the market right now.",
    "Volume (24h)": "Total value traded in the last 24 hours - shows how active the coin is.",
    "ATH": "All-Time High - the highest price the coin has ever reached.",
    "ATL": "All-Time Low - the lowest price the coin has ever reached.",
    "Liquidity": "How easily you can buy/sell without moving the price much.",
    "OHLCV": "Open, High, Low, Close, Volume - the standard candle data for each time period.",
    "Stop-loss": "A preset exit price that caps your loss if the trade goes against you.",
    "Take-profit": "A preset exit price where you lock in gains if the trade goes your way.",
    "Expectancy": "Average profit or loss per trade over many trades, after costs.",
    "Meta-label probability": "A second model's estimate of the chance your entry actually hits take-profit before stop-loss.",
    "ATR": "Average True Range - how much the price typically moves per bar.",
    "Triple-barrier": "Labels a trade by what comes first: take-profit, stop-loss, or a time limit.",
    "Kelly": "A formula for bet size that maximizes long-run growth; use a small fraction of it to cut risk.",
    "Breakout": "Price pushing past a recent high/low, often continuing in that direction.",
    "Dip-buy": "Buying after a pullback toward support, expecting a bounce (range markets).",
    "Regime": "Whether the market is currently trending or range-bound (chopping sideways).",
}


def normalize(term: str) -> str:
    """Lower-case, collapse whitespace, trim — for case/spacing-insensitive lookup."""
    return re.sub(r"\s+", " ", str(term or "").strip().lower())


_NORM: Dict[str, str] = {normalize(k): v for k, v in GLOSSARY.items()}


def get_definition(term: str) -> Optional[str]:
    """Definition for a term (case-insensitive, plural-tolerant), or None."""
    key = normalize(term)
    if not key:
        return None
    if key in _NORM:
        return _NORM[key]
    if key.endswith("s") and key[:-1] in _NORM:   # plural -> singular ("RSIs" -> "RSI")
        return _NORM[key[:-1]]
    if (key + "s") in _NORM:                        # singular -> plural ("Bollinger Band")
        return _NORM[key + "s"]
    return None


def terms() -> List[str]:
    """All known display terms (used to build the auto-annotate matcher)."""
    return list(GLOSSARY.keys())
