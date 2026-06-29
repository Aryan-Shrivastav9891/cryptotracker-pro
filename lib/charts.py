"""Plotly chart helpers — the Python replacement for the React Chart.js charts.

Produces smooth, interactive, theme-aware line charts with the same
green/red price-trend styling and ₹ tooltips as the original.
"""
from __future__ import annotations

from typing import List, Optional

import plotly.graph_objects as go

GREEN = "#34d399"
RED = "#ef4444"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _style(fig: go.Figure, height: int, dark: bool) -> None:
    grid = "rgba(255,255,255,0.06)" if dark else "rgba(0,0,0,0.06)"
    text = "#e5e7eb" if dark else "#4b5563"
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text),
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, color=text)
    fig.update_yaxes(showgrid=True, gridcolor=grid, color=text, tickprefix="₹", tickformat=",.0f")


def line_chart(
    prices: List[float],
    labels: Optional[List[str]] = None,
    positive: bool = True,
    height: int = 320,
    dark: bool = False,
) -> go.Figure:
    color = GREEN if positive else RED
    x = labels if labels is not None else list(range(len(prices)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=prices,
            mode="lines",
            line=dict(color=color, width=2, shape="spline"),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, 0.15),
            hovertemplate="₹%{y:,.2f}<extra></extra>",
        )
    )
    _style(fig, height, dark)
    return fig


def forecast_interval_chart(
    history_dates: List,
    history_prices: List[float],
    forecast_dates: List,
    forecast_prices: List[float],
    lower: List[float],
    upper: List[float],
    positive: bool = True,
    height: int = 420,
    dark: bool = False,
) -> go.Figure:
    """History + multi-step forecast with an 80% confidence band."""
    color = GREEN if positive else RED
    hist_color = "#60a5fa" if dark else "#2563eb"
    text = "#e5e7eb" if dark else "#4b5563"
    grid = "rgba(255,255,255,0.06)" if dark else "rgba(0,0,0,0.06)"

    fig = go.Figure()
    # Confidence band (upper, then lower with fill between).
    fig.add_trace(
        go.Scatter(x=forecast_dates, y=upper, mode="lines", line=dict(width=0),
                   showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(x=forecast_dates, y=lower, mode="lines", line=dict(width=0),
                   fill="tonexty", fillcolor=_hex_to_rgba(color, 0.18),
                   name="80% interval", hoverinfo="skip")
    )
    # History.
    fig.add_trace(
        go.Scatter(x=history_dates, y=history_prices, mode="lines",
                   line=dict(color=hist_color, width=2), name="History",
                   hovertemplate="₹%{y:,.2f}<extra></extra>")
    )
    # Forecast (prepend last history point for a continuous line).
    fx = ([history_dates[-1]] + list(forecast_dates)) if history_dates else list(forecast_dates)
    fy = ([history_prices[-1]] + list(forecast_prices)) if history_prices else list(forecast_prices)
    fig.add_trace(
        go.Scatter(x=fx, y=fy, mode="lines+markers",
                   line=dict(color=color, width=2, dash="dash"),
                   marker=dict(size=5, color=color), name="Forecast",
                   hovertemplate="Forecast ₹%{y:,.2f}<extra></extra>")
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, color=text)
    fig.update_yaxes(showgrid=True, gridcolor=grid, color=text, tickprefix="₹", tickformat=",.0f")
    return fig
