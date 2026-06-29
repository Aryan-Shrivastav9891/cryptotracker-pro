"""Shared Streamlit UI: theme, header, sidebar controls, market overview,
coin cards and footer. Ports the React components (Header, Tabs, MarketOverview,
CoinCard, Footer) into reusable Streamlit helpers.
"""
from __future__ import annotations

import html
import json
import re
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from lib import glossary, model
from lib.formatting import (
    change_color,
    format_inr,
    format_pct,
    human_format,
    safe_float,
)

COIN_DETAILS_PAGE = "views/coin_details.py"


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    st.session_state.setdefault("search", "")
    st.session_state.setdefault("dark_mode", False)
    st.session_state.setdefault("selected_coin_id", None)
    st.session_state.setdefault("show_count", 30)


# --------------------------------------------------------------------------- #
# Theming (runtime dark mode via injected CSS)
# --------------------------------------------------------------------------- #
def tokens(dark: bool) -> Dict[str, str]:
    """Single source of truth for the design palette (used by CSS + components)."""
    if dark:
        return {
            "appbg": "radial-gradient(1100px 560px at 18% -12%, #15213b 0%, #0b1020 58%)",
            "surface": "rgba(30,41,59,0.55)", "surface_solid": "#0f1830",
            "border": "rgba(148,163,184,0.18)", "text": "#e6eaf2",
            "muted": "#94a3b8", "accent": "#8b5cf6", "secondary": "#22d3ee",
            "success": "#34d399", "danger": "#f87171", "sidebar": "#0b1228",
            "shadow": "0 10px 30px rgba(0,0,0,0.45)", "shadow_hover": "0 16px 42px rgba(0,0,0,0.55)",
        }
    return {
        "appbg": "radial-gradient(1100px 560px at 18% -12%, #eef2ff 0%, #f4f6fb 58%)",
        "surface": "rgba(255,255,255,0.72)", "surface_solid": "#ffffff",
        "border": "rgba(15,23,42,0.08)", "text": "#0f172a",
        "muted": "#5b6678", "accent": "#7c3aed", "secondary": "#0891b2",
        "success": "#059669", "danger": "#dc2626", "sidebar": "#ffffff",
        "shadow": "0 10px 24px rgba(2,6,23,0.08)", "shadow_hover": "0 16px 36px rgba(2,6,23,0.14)",
    }


_FONTS = ("<style>@import url('https://fonts.googleapis.com/css2?"
          "family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');</style>")

_STATIC_CSS = """
.stApp { background: var(--bg); transition: background .4s ease, color .4s ease; }
.stApp, .stApp p, .stApp span, .stApp li, .stApp label {
    color: var(--text); font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
h1, h2, h3, h4 { font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -.01em; color: var(--text); }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', 'Inter', sans-serif; color: var(--text); }
[data-testid="stMetricLabel"] { color: var(--muted); }
/* glassmorphism cards (Streamlit bordered containers) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
    box-shadow: var(--shadow); transition: transform .18s ease, box-shadow .18s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px); box-shadow: var(--shadow-hover);
}
.stButton > button { border-radius: 12px; border: 1px solid var(--border); transition: all .15s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: var(--shadow); border-color: var(--accent); }
@keyframes ctp-shimmer { 0% { background-position: -468px 0 } 100% { background-position: 468px 0 } }
.ctp-skeleton {
    background: linear-gradient(90deg, var(--surface) 25%, rgba(148,163,184,.18) 37%, var(--surface) 63%);
    background-size: 936px 100%; animation: ctp-shimmer 1.5s infinite linear; border-radius: 12px;
}
.ctp-brand {
    font-size: 2.3rem; font-weight: 700; margin-bottom: 0;
    background: linear-gradient(90deg, var(--accent), var(--secondary));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.ctp-muted { color: var(--muted) !important; }
.ctp-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem 1.15rem; }
.ctp-badge { padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.ctp-pill { padding: 2px 8px; border-radius: 999px; font-size: .72rem; }
.gloss {
    border-bottom: 1px dotted var(--accent); background: rgba(124,58,237,.12);
    border-radius: 3px; padding: 0 2px; cursor: help; position: relative;
}
.gloss::after {
    content: attr(data-tip); position: absolute; left: 50%; bottom: 150%; transform: translateX(-50%);
    width: 240px; max-width: 80vw; white-space: normal; text-align: left;
    background: #0f172a; color: #f9fafb; border: 1px solid var(--accent); border-radius: 8px;
    padding: 8px 10px; font-size: .78rem; line-height: 1.35; box-shadow: 0 6px 20px rgba(0,0,0,.35);
    opacity: 0; visibility: hidden; transition: opacity .15s ease; z-index: 9999; pointer-events: none;
}
.gloss:hover::after { opacity: 1; visibility: visible; }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .001ms !important; animation-iteration-count: 1 !important;
        transition-duration: .001ms !important; }
}
"""


def apply_theme(dark: bool) -> None:
    """Inject the full design system (tokens + glass cards + fonts) for the active theme.

    Emitted every rerun (cheap, idempotent by position) so the dark/light toggle
    updates instantly; runtime-heavy animation lives in the anime.js components.
    """
    t = tokens(dark)
    css_vars = (
        ":root{"
        f"--bg:{t['appbg']};--surface:{t['surface']};--border:{t['border']};"
        f"--text:{t['text']};--muted:{t['muted']};--accent:{t['accent']};"
        f"--secondary:{t['secondary']};--success:{t['success']};--danger:{t['danger']};"
        f"--shadow:{t['shadow']};--shadow-hover:{t['shadow_hover']};--radius:18px;"
        "}"
    )
    sidebar = f"[data-testid='stSidebar']{{background:{t['sidebar']};}}"
    st.markdown(_FONTS + "<style>" + css_vars + _STATIC_CSS + sidebar + "</style>",
                unsafe_allow_html=True)


inject_theme = apply_theme  # alias per the design-system API


# --------------------------------------------------------------------------- #
# anime.js components (self-contained iframes; the ONLY place JS animation lives)
# --------------------------------------------------------------------------- #
_ANIME_CDN = "https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"

_BENTO_CSS = """
*{box-sizing:border-box;} html,body{margin:0;background:transparent;}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;color:var(--text);}
.bento{display:grid;gap:14px;grid-template-columns:repeat(4,1fr);
  grid-template-areas:"hero hero sig sig" "hero hero k1 k2" "hero hero k3 k4";}
.cell{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:14px 16px;
  box-shadow:var(--shadow);opacity:0;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);}
.lab{font-size:.70rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);}
.hero{grid-area:hero;display:flex;flex-direction:column;justify-content:center;min-height:178px;position:relative;overflow:hidden;}
.hero .big{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:2.5rem;line-height:1.05;color:var(--text);}
.hero .delta{font-weight:600;margin-top:2px;}
.hero .spark{width:100%;height:48px;margin-top:10px;}
.hero::after{content:"";position:absolute;top:0;left:-60%;width:40%;height:100%;
  background:linear-gradient(120deg,transparent,rgba(255,255,255,.20),transparent);transform:skewX(-20deg);transition:left .6s ease;}
.hero:hover{transform:scale(1.012);transition:transform .2s ease;}
.hero:hover::after{left:130%;}
.kpi{display:flex;flex-direction:column;justify-content:center;}
.kpi .val{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.55rem;color:var(--text);margin-top:3px;}
.sig{grid-area:sig;display:flex;flex-direction:column;gap:8px;justify-content:center;}
.badge{display:inline-block;align-self:flex-start;padding:6px 16px;border-radius:999px;font-weight:700;font-size:1.05rem;border:1.5px solid;}
.badge.buy{color:var(--success);border-color:var(--success);animation:pulseG 1.9s infinite;}
.badge.sell{color:var(--danger);border-color:var(--danger);animation:pulseR 1.9s infinite;}
.badge.hold,.badge.none{color:var(--muted);border-color:var(--border);}
@keyframes pulseG{0%{box-shadow:0 0 0 0 rgba(52,211,153,.45)}70%{box-shadow:0 0 0 13px rgba(52,211,153,0)}100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}
@keyframes pulseR{0%{box-shadow:0 0 0 0 rgba(248,113,113,.45)}70%{box-shadow:0 0 0 13px rgba(248,113,113,0)}100%{box-shadow:0 0 0 0 rgba(248,113,113,0)}}
@media (max-width:680px){.bento{grid-template-columns:1fr 1fr;grid-template-areas:"hero hero" "sig sig" "k1 k2" "k3 k4";}}
@media (prefers-reduced-motion: reduce){.cell{opacity:1!important;}.badge{animation:none!important;}.hero::after{display:none;}}
"""

_BENTO_JS = """
const C = window.CFG;
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function fmt(v, kind, dec, signed){
  dec = (dec==null)?2:dec; v = Number(v);
  if(kind==='pct'){ return (signed&&v>=0?'+':'')+v.toFixed(dec)+'%'; }
  if(kind==='inr'){ return '₹'+new Intl.NumberFormat('en-IN',{maximumFractionDigits:dec}).format(v); }
  if(kind==='compact'){ return '₹'+new Intl.NumberFormat('en-IN',{notation:'compact',maximumFractionDigits:dec}).format(v); }
  return new Intl.NumberFormat('en-IN',{notation:'compact',maximumFractionDigits:dec}).format(v);
}
const grid = document.getElementById('bento');
const hero = document.createElement('div'); hero.className='cell hero';
hero.innerHTML = '<div class="lab">'+C.hero.label+'</div>'+
  '<div class="big"><span id="heroVal">0</span></div>'+
  '<div class="delta" id="heroDelta"></div>'+
  '<svg class="spark" viewBox="0 0 100 30" preserveAspectRatio="none"><path id="sparkPath" fill="none" stroke-linejoin="round"/></svg>';
grid.appendChild(hero);
const sig = document.createElement('div'); sig.className='cell sig';
sig.innerHTML = '<div class="lab">Signal</div><div class="badge '+C.signal.kind+'">'+C.signal.text+'</div>';
grid.appendChild(sig);
C.kpis.forEach(function(k,i){ const d=document.createElement('div'); d.className='cell kpi'; d.style.gridArea='k'+(i+1);
  d.innerHTML='<div class="lab" title="'+(k.tip||'')+'">'+k.label+'</div><div class="val"><span id="kv'+i+'">0</span></div>';
  grid.appendChild(d); });
// delta text
if(C.hero.delta!=null){ const de=document.getElementById('heroDelta'); const up=C.hero.delta>=0;
  de.textContent=(up?'▲ +':'▼ ')+Math.abs(C.hero.delta).toFixed(2)+'%'; de.style.color=up?C.col.success:C.col.danger; }
// sparkline path
const sp=C.hero.spark; const path=document.getElementById('sparkPath');
if(sp && sp.length>1){ const mn=Math.min.apply(null,sp), mx=Math.max.apply(null,sp), rng=(mx-mn)||1;
  let d='M'; sp.forEach(function(v,i){ d+=(i*100/(sp.length-1)).toFixed(2)+' '+(28-((v-mn)/rng)*26).toFixed(2)+(i<sp.length-1?' L ':''); });
  path.setAttribute('d',d); path.setAttribute('stroke', sp[sp.length-1]>=sp[0]?C.col.success:C.col.danger); path.setAttribute('stroke-width','1.8');
} else if(path){ path.parentNode.style.display='none'; }
function setFinals(){
  document.getElementById('heroVal').textContent=fmt(C.hero.value,C.hero.fmt,C.hero.decimals,false);
  C.kpis.forEach(function(k,i){ document.getElementById('kv'+i).textContent=fmt(k.value,k.fmt,k.decimals,k.signed); });
  document.querySelectorAll('.cell').forEach(function(c){c.style.opacity=1;});
  if(path&&path.getTotalLength){ path.style.strokeDasharray='none'; }
}
function animate(){
  document.querySelectorAll('.cell').forEach(function(c){c.style.opacity=0;});
  anime({targets:'.cell', translateY:[18,0], opacity:[0,1], delay:anime.stagger(80), duration:600, easing:'easeOutCubic'});
  function countUp(id,target,kind,dec,signed){ const el=document.getElementById(id); const o={v:0};
    anime({targets:o, v:target, duration:1500, easing:'easeOutExpo', update:function(){ el.textContent=fmt(o.v,kind,dec,signed); }}); }
  countUp('heroVal',C.hero.value,C.hero.fmt,C.hero.decimals,false);
  C.kpis.forEach(function(k,i){ countUp('kv'+i,k.value,k.fmt,k.decimals,k.signed); });
  if(path&&path.getTotalLength){ const len=path.getTotalLength(); path.style.strokeDasharray=len; path.style.strokeDashoffset=len;
    anime({targets:path, strokeDashoffset:[len,0], duration:1400, easing:'easeInOutSine', delay:300}); }
}
if(reduce || typeof anime==='undefined'){ setFinals(); } else { animate(); }
"""


def _signal_kind(signal: str) -> str:
    s = (signal or "").lower()
    if "buy" in s:
        return "buy"
    if "sell" in s:
        return "sell"
    if "edge" in s or s in ("", "n/a"):
        return "none"
    return "hold"


def animated_bento(dark: bool, hero: Dict[str, Any], kpis: List[Dict[str, Any]],
                   signal: Dict[str, str], height: int = 250) -> None:
    """Animated bento grid (hero + KPI cells + signal badge) rendered via anime.js.

    hero: {label, value, fmt('inr'|'compact'|'num'), decimals, delta(%|None), spark([...]|None)}
    kpis: up to 4 dicts {label, value, fmt('pct'|'num'|'inr'|'compact'), decimals, signed, tip}
    signal: {text, kind('buy'|'sell'|'hold'|'none')}
    """
    t = tokens(dark)
    payload = {
        "hero": hero,
        "kpis": kpis[:4],
        "signal": signal,
        "col": {"success": t["success"], "danger": t["danger"]},
    }
    root = (":root{"
            f"--text:{t['text']};--muted:{t['muted']};--surface:{t['surface']};"
            f"--border:{t['border']};--shadow:{t['shadow']};--accent:{t['accent']};"
            f"--success:{t['success']};--danger:{t['danger']};" "}")
    head = ("<style>@import url('https://fonts.googleapis.com/css2?"
            "family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');"
            + root + _BENTO_CSS + "</style>")
    body = ('<div id="bento" class="bento"></div>'
            f"<script>window.CFG={json.dumps(payload)};</script>"
            f'<script src="{_ANIME_CDN}"></script>'
            f"<script>{_BENTO_JS}</script>")
    components.html("<!doctype html><html><head>" + head + "</head><body>" + body + "</body></html>",
                    height=height, scrolling=False)


def signal_badge(signal: str, dark: bool, height: int = 64) -> None:
    """A standalone pulsing signal badge (anime.js / CSS), colour-coded by signal."""
    t = tokens(dark)
    kind = _signal_kind(signal)
    color = {"buy": t["success"], "sell": t["danger"]}.get(kind, t["muted"])
    css = ("*{box-sizing:border-box}body{margin:0;background:transparent;"
           "font-family:'Inter',system-ui,sans-serif;display:flex;align-items:center;}"
           ".b{padding:6px 16px;border-radius:999px;font-weight:700;border:1.5px solid " + color
           + ";color:" + color + ";opacity:0;}"
           "@keyframes pz{0%{box-shadow:0 0 0 0 " + color + "55}70%{box-shadow:0 0 0 12px " + color
           + "00}100%{box-shadow:0 0 0 0 " + color + "00}}"
           "@media (prefers-reduced-motion: reduce){.b{animation:none!important;opacity:1!important}}")
    js = ("var b=document.querySelector('.b');var r=window.matchMedia('(prefers-reduced-motion: reduce)').matches;"
          "if(r||typeof anime==='undefined'){b.style.opacity=1;}else{"
          "anime({targets:b,opacity:[0,1],scale:[0.9,1],duration:500,easing:'easeOutBack'});"
          "b.style.animation='pz 1.9s infinite';}")
    html_doc = ("<!doctype html><html><head><style>" + css + "</style></head><body>"
                f'<span class="b">{html.escape(signal)}</span>'
                f'<script src="{_ANIME_CDN}"></script><script>{js}</script></body></html>')
    components.html(html_doc, height=height, scrolling=False)


def kpi_card(label: str, value: str, help: Optional[str] = None) -> None:
    """Native metric KPI with a glossary tooltip on the label (non-animated contexts)."""
    st.metric(label, value, help=help or glossary.get_definition(label))


def bento_columns(weights: List[int]):
    """Structural helper: columns whose bordered containers become glass bento cells."""
    return st.columns(weights, gap="medium")


# --------------------------------------------------------------------------- #
# Glossary tooltips (inline hover help for jargon)
# --------------------------------------------------------------------------- #
_terms_sorted = sorted(glossary.terms(), key=len, reverse=True)
_GLOSS_PATTERN = (
    re.compile(r"(?<![\w])(" + "|".join(re.escape(t) for t in _terms_sorted) + r")s?(?![\w])",
               re.IGNORECASE)
    if _terms_sorted else None
)


def term(label: str, definition: str = None) -> str:
    """Return an HTML span that shows ``definition`` on hover (with native title fallback).

    If ``definition`` is omitted it's looked up in the glossary. Unknown terms are
    returned as plain (escaped) text. Render with st.markdown(..., unsafe_allow_html=True).
    """
    if definition is None:
        definition = glossary.get_definition(label)
    lab = html.escape(str(label))
    if not definition:
        return lab
    tip = html.escape(definition, quote=True)
    return f'<span class="gloss" data-tip="{tip}" title="{tip}">{lab}</span>'


def _wrap_match(match: "re.Match", used: set) -> str:
    matched = match.group(0)
    definition = glossary.get_definition(matched)
    if not definition:
        return matched
    key = glossary.normalize(matched).rstrip("s")
    if key in used:  # only the first occurrence of each term per text block
        return matched
    used.add(key)
    return term(matched, definition)


def annotate(text: str) -> str:
    """Auto-wrap the first occurrence of each known glossary term in a hover tooltip.

    Skips text inside `backtick code spans`, never double-wraps, and matches
    whole words case-insensitively. Render with st.markdown(..., unsafe_allow_html=True).
    """
    if not text or _GLOSS_PATTERN is None:
        return text
    used: set = set()
    out: List[str] = []
    for part in re.split(r"(`[^`]*`)", str(text)):  # keep code spans verbatim
        if part[:1] == "`":
            out.append(part)
        else:
            out.append(_GLOSS_PATTERN.sub(lambda m: _wrap_match(m, used), part))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Header / sidebar
# --------------------------------------------------------------------------- #
def top_bar() -> None:
    st.markdown('<p class="ctp-brand">CryptoTracker Pro</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="ctp-muted">Track crypto prices, AI predictions and market '
        "insights — now powered by Python.</p>",
        unsafe_allow_html=True,
    )


def sidebar_controls() -> None:
    """Render the persistent sidebar (search + dark mode + model + refresh)."""
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.text_input("🔎 Search coins", key="search", placeholder="Bitcoin, ETH…")
        st.toggle("🌙 Dark mode", key="dark_mode")
        st.divider()
        st.caption(model.model_status())
        if st.button("🔄 Refresh live data", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Data: CoinGecko · News: CryptoCompare")


# --------------------------------------------------------------------------- #
# Navigation helper
# --------------------------------------------------------------------------- #
def go_to_coin(coin_id: str) -> None:
    st.session_state.selected_coin_id = coin_id
    st.switch_page(COIN_DETAILS_PAGE)


# --------------------------------------------------------------------------- #
# Market overview (ports MarketOverview.js)
# --------------------------------------------------------------------------- #
def market_overview(coins: List[Dict[str, Any]]) -> None:
    if not coins:
        return
    total_mc = sum(safe_float(c.get("market_cap")) for c in coins)
    total_vol = sum(safe_float(c.get("total_volume")) for c in coins)
    changes = [safe_float(c.get("price_change_percentage_24h")) for c in coins]
    avg_change = sum(changes) / len(changes) if changes else 0.0
    bullish = avg_change > 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Status", "Bullish 🐂" if bullish else "Bearish 🐻")
    c2.metric("Avg 24h Change", format_pct(avg_change), delta=f"{avg_change:.2f}%")
    c3.metric("Total Market Cap", human_format(total_mc, prefix="₹"))
    c4.metric("24h Volume", human_format(total_vol, prefix="₹"))

    movers = sorted(
        coins,
        key=lambda c: abs(safe_float(c.get("price_change_percentage_24h"))),
        reverse=True,
    )[:3]
    if movers:
        chips = "  ".join(
            f"**{m.get('symbol', '').upper()}** "
            f"{format_pct(m.get('price_change_percentage_24h'), signed=True)}"
            for m in movers
        )
        st.caption(f"🔥 Top movers: {chips}")


# --------------------------------------------------------------------------- #
# Coin card (ports CoinCard.js)
# --------------------------------------------------------------------------- #
def _sell_advice(change_24h: float) -> Dict[str, str]:
    if change_24h < -10:
        return {
            "label": "⚠️ Sell Alert",
            "text": "Price dropped significantly in the last 24h — consider selling.",
            "bg": "rgba(239,68,68,0.12)",
            "fg": "#ef4444",
        }
    if change_24h > 20:
        return {
            "label": "💰 Profit Opportunity",
            "text": "Price rose significantly — consider locking in profits.",
            "bg": "rgba(234,179,8,0.14)",
            "fg": "#ca8a04",
        }
    return {
        "label": "📈 Market Insight",
        "text": "No immediate sell signal. Monitor the market for changes.",
        "bg": "rgba(59,130,246,0.12)",
        "fg": "#3b82f6",
    }


def coin_card(coin: Dict[str, Any], dark: bool, key: str) -> None:
    change = safe_float(coin.get("price_change_percentage_24h"))
    color = change_color(change, dark)
    arrow = "▲" if change >= 0 else "▼"
    advice = _sell_advice(change)
    rank = coin.get("market_cap_rank") or "N/A"

    with st.container(border=True):
        head = st.columns([1, 4, 2])
        with head[0]:
            if coin.get("image"):
                st.image(coin["image"], width=40)
        with head[1]:
            star = " ⭐" if (coin.get("market_cap_rank") or 999) <= 10 else ""
            st.markdown(
                f"**{coin.get('name', '?')}**{star}  \n"
                f"<span class='ctp-muted'>{coin.get('symbol', '').upper()} · "
                f"Rank #{rank}</span>",
                unsafe_allow_html=True,
            )
        with head[2]:
            st.markdown(
                f"<div style='text-align:right'>"
                f"<span class='ctp-badge' style='background:{advice['bg']};color:{color}'>"
                f"{arrow} {abs(change):.2f}%</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div style='font-size:1.6rem;font-weight:800'>"
            f"{format_inr(coin.get('current_price'))}</div>",
            unsafe_allow_html=True,
        )

        if coin.get("predicted_price") is not None:
            pred_pct = safe_float(coin.get("predicted_change"))
            st.markdown(
                f"<span class='ctp-muted'>AI predicted: </span>"
                f"<b>{format_inr(coin.get('predicted_price'))}</b> "
                f"<span style='color:{change_color(pred_pct, dark)}'>"
                f"({format_pct(pred_pct, signed=True)})</span>",
                unsafe_allow_html=True,
            )

        if coin.get("description"):
            desc = str(coin["description"])
            preview = desc[:100] + ("…" if len(desc) > 100 else "")
            st.markdown(
                f"<div class='ctp-muted' style='font-size:0.84rem;margin-top:0.3rem'>"
                f"{preview}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div class='ctp-card' style='background:{advice['bg']};"
            f"border:none;padding:0.6rem 0.8rem;margin-top:0.4rem'>"
            f"<b style='color:{advice['fg']}'>{advice['label']}</b><br>"
            f"<span style='font-size:0.82rem'>{advice['text']}</span></div>",
            unsafe_allow_html=True,
        )

        if st.button("View details →", key=key, width="stretch"):
            go_to_coin(coin.get("id"))


def coin_grid(coins: List[Dict[str, Any]], dark: bool, key_prefix: str, columns: int = 3) -> None:
    if not coins:
        st.info("No coins match your criteria. Try adjusting your search.")
        return
    cols = st.columns(columns)
    for i, coin in enumerate(coins):
        with cols[i % columns]:
            coin_card(coin, dark, key=f"{key_prefix}_{coin.get('id', i)}")


# --------------------------------------------------------------------------- #
# Footer (ports Footer.js)
# --------------------------------------------------------------------------- #
def footer() -> None:
    st.divider()
    st.markdown(
        "<div class='ctp-muted' style='text-align:center;font-size:0.85rem'>"
        "<b>CryptoTracker Pro</b> · Track crypto prices, get market insights, "
        "and make better trading decisions.<br>"
        "© 2026 CryptoTracker Pro. All rights reserved. · For educational use only — "
        "not financial advice.</div>",
        unsafe_allow_html=True,
    )
