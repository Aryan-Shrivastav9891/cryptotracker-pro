"""Number, currency and percentage formatting helpers.

Faithful to the original app which used `Intl.NumberFormat('en-IN', ...)` —
i.e. Indian digit grouping (e.g. 12,34,567) and rupee prefixes, plus a
compact (K/M/B/T) notation for very large values.
"""
from __future__ import annotations

import math
from datetime import datetime
from html.parser import HTMLParser
from typing import Optional


def safe_float(value, default: float = 0.0) -> float:
    """Coerce anything to float, returning ``default`` on failure/None/NaN/±inf."""
    try:
        if value is None:
            return default
        f = float(value)
        if not math.isfinite(f):  # reject NaN and ±inf
            return default
        return f
    except (TypeError, ValueError):
        return default


def _indian_group(integer_str: str) -> str:
    """Group an integer string the Indian way: last 3 digits, then pairs."""
    negative = integer_str.startswith("-")
    if negative:
        integer_str = integer_str[1:]

    if len(integer_str) <= 3:
        grouped = integer_str
    else:
        last3 = integer_str[-3:]
        rest = integer_str[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3

    return ("-" + grouped) if negative else grouped


def format_number_in(value, decimals: int = 2) -> str:
    """Indian-grouped number, e.g. 1234567.5 -> '12,34,567.5'."""
    if value is None:
        return "N/A"
    f = float(value) if isinstance(value, (int, float)) else safe_float(value, default=None)
    if f is None or not math.isfinite(f):
        return "N/A"

    negative = f < 0
    f = abs(f)
    quantized = f"{f:.{decimals}f}"
    if "." in quantized:
        int_part, frac_part = quantized.split(".")
        frac_part = frac_part.rstrip("0")
    else:
        int_part, frac_part = quantized, ""

    grouped = _indian_group(int_part)
    out = grouped + (("." + frac_part) if frac_part else "")
    return ("-" + out) if negative else out


def human_format(num, prefix: str = "") -> str:
    """Compact notation: 1.23B, 4.56M, etc. Mirrors `notation: 'compact'`."""
    if num is None:
        return "N/A"
    n = safe_float(num, default=None)
    if n is None:
        return "N/A"

    negative = n < 0
    n = abs(n)
    for unit in ["", "K", "M", "B", "T"]:
        if n < 1000:
            # Drop trailing ".0" for whole numbers, keep up to 2 decimals.
            text = f"{n:.2f}".rstrip("0").rstrip(".")
            value = f"{prefix}{text}{unit}"
            return ("-" + value) if negative else value
        n /= 1000.0
    value = f"{prefix}{n:.2f}Q"  # quadrillion fallback
    return ("-" + value) if negative else value


def format_inr(value, decimals: int = 2) -> str:
    """Rupee-prefixed Indian-grouped amount, e.g. '₹12,34,567.50'."""
    if value is None:
        return "N/A"
    body = format_number_in(value, decimals=decimals)
    return "N/A" if body == "N/A" else "₹" + body


def format_inr_compact(value) -> str:
    """Rupee amount in compact form when large, otherwise full Indian grouping.

    Mirrors the original `formatNumber`: compact above 1,000,000.
    """
    f = safe_float(value, default=None) if value is not None else None
    if f is None:
        return "N/A"
    if abs(f) > 1_000_000:
        return human_format(f, prefix="₹")
    return format_inr(f)


def format_pct(value, decimals: int = 2, signed: bool = False) -> str:
    """Percentage string, e.g. '+12.34%' or '5.00%'."""
    f = safe_float(value, default=None)
    if f is None:  # None, NaN or ±inf
        return "N/A"
    sign = "+" if (signed and f >= 0) else ""
    return f"{sign}{f:.{decimals}f}%"


def change_color(value: Optional[float], dark: bool = False) -> str:
    """Green for non-negative change, red for negative — as hex for inline CSS."""
    f = safe_float(value)
    if f >= 0:
        return "#34d399" if dark else "#10b981"  # green-400 / emerald-500
    return "#f87171" if dark else "#ef4444"        # red-400 / red-500


def format_date(value, fmt: str = "%d %b %Y") -> str:
    """Format an ISO-8601 timestamp (e.g. CoinGecko ath_date) as a date string.

    Robust on Python 3.9: parses just the leading ``YYYY-MM-DD``. Returns "" on
    anything it can't parse.
    """
    if not value:
        return ""
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").strftime(fmt)
    except (ValueError, TypeError):
        return ""


class _HTMLStripper(HTMLParser):
    _SKIP = {"script", "style"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def strip_html(text: Optional[str]) -> str:
    """Strip HTML tags from untrusted text (e.g. CoinGecko descriptions).

    Avoids rendering untrusted third-party HTML (XSS-safe) while keeping the
    readable text. Returns "" for falsy input.
    """
    if not text:
        return ""
    try:
        stripper = _HTMLStripper()
        stripper.feed(str(text))
        return stripper.text().strip()
    except Exception:
        return str(text)
