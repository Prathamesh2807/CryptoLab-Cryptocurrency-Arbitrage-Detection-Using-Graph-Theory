"""
CryptoLab — Cryptocurrency Arbitrage Detection Dashboard

An interactive Discrete Mathematics simulation and research laboratory
demonstrating cryptocurrency arbitrage detection using directed graph theory,
negative-weight cycle detection (Bellman-Ford algorithm), and transaction-fee simulation.

DISCLAIMER: All exchange rates and scenarios in this application are illustrative,
simulated data for educational purposes only. No live market feeds, wallets,
or real financial trading are involved.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

import plotly.graph_objects as go
import streamlit as st

from src.arbitrage_detector import ArbitrageResult, detect_arbitrage
from src.exchange_rates import (
    SIMULATION_DISCLAIMER,
    ConversionRate,
    get_arbitrage_sample_rates,
    get_fee_consumed_sample_rates,
    get_inr_arbitrage_sample_rates,
    get_sample_rates,
    validate_conversion_rate,
)
from src.graph_builder import build_graph, calculate_log_weight
from src.profit_calculator import ProfitResult, calculate_profit


# ==============================================================================
# Helper Functions: Data Formatting & Resilient Table Display
# ==============================================================================

def render_table(headers: List[str], rows: List[List[Any]]) -> None:
    """
    Render a clean, high-contrast financial-style HTML table.
    Styled according to the Midnight Navy + Electric Cyan palette.
    """
    header_html = "".join(
        f'<th style="padding: 11px 16px; text-align: left; font-weight: 600; '
        f'color: #06B6D4; background-color: #162032; text-transform: uppercase; '
        f'font-size: 0.76rem; letter-spacing: 0.05em; border-bottom: 1px solid #26364C;">{h}</th>'
        for h in headers
    )

    body_rows: List[str] = []
    for idx, row in enumerate(rows):
        bg = 'background-color: rgba(22, 32, 50, 0.4);' if idx % 2 == 1 else 'background-color: transparent;'
        cells = "".join(
            f'<td style="padding: 10px 16px; border-bottom: 1px solid rgba(38, 54, 76, 0.5); '
            f'color: #E2E8F0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; '
            f'font-size: 0.88rem;">{cell}</td>'
            for cell in row
        )
        body_rows.append(f'<tr style="{bg}" class="table-row-hover">{cells}</tr>')

    table_html = (
        '<div style="overflow-x: auto; margin: 14px 0; border-radius: 10px; '
        'border: 1px solid #26364C; background-color: #0F172A; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);">'
        '<table style="width: 100%; border-collapse: collapse; font-family: -apple-system, '
        'BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ==============================================================================
# Helper Functions: Plotly Directed Graph Visualization
# ==============================================================================

def create_directed_network_figure(
    graph: Any,
    highlight_cycle: Optional[List[str]] = None,
) -> go.Figure:
    """
    Generate an interactive Plotly visualization for the directed currency graph.
    Themed with Midnight Navy (#070B16), Electric Cyan (#06B6D4), and Vibrant Green (#34D399).

    - Vertices arranged in a circular topology with distinct currency badges.
    - Directed curved edges with distinct arrow annotations indicating conversion direction.
    - Rate annotations positioned at the apex of each edge.
    - Cycle edges and vertices prominently highlighted in emerald green.
    """
    fig = go.Figure()

    if graph is None or graph.number_of_nodes() == 0:
        fig.add_annotation(
            text="Currency graph is empty. Please add exchange rate pairs to visualize the network.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#94A3B8"),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=460,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        return fig

    nodes = sorted(list(graph.nodes()))
    num_nodes = len(nodes)

    # Compute 2D coordinates using circular layout
    coords: Dict[str, Tuple[float, float]] = {}
    radius = 1.0
    for idx, node in enumerate(nodes):
        angle = 2.0 * math.pi * idx / num_nodes
        coords[node] = (radius * math.cos(angle), radius * math.sin(angle))

    # Identify cycle edges for accent highlighting
    cycle_edge_set = set()
    if highlight_cycle and len(highlight_cycle) >= 3:
        for i in range(len(highlight_cycle) - 1):
            cycle_edge_set.add((highlight_cycle[i], highlight_cycle[i + 1]))

    # Draw directed edges
    annotations = []
    for u, v, data in graph.edges(data=True):
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        rate = data.get("rate", 1.0)
        weight = data.get("weight", 0.0)

        is_cycle_edge = (u, v) in cycle_edge_set

        # Color and line width matching theme
        edge_color = "#34D399" if is_cycle_edge else "rgba(6, 182, 212, 0.45)"
        edge_width = 3.8 if is_cycle_edge else 1.8

        # Compute curvature so bidirectional edges separate cleanly
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue

        # Perpendicular unit normal vector
        nx = -dy / dist
        ny = dx / dist
        curvature_offset = 0.18
        xc = (x0 + x1) / 2.0 + curvature_offset * nx
        yc = (y0 + y1) / 2.0 + curvature_offset * ny

        # Quadratic Bezier curve points
        curve_pts = 26
        curve_x: List[float] = []
        curve_y: List[float] = []
        for step in range(curve_pts + 1):
            t = step / curve_pts
            bx = ((1.0 - t) ** 2) * x0 + 2.0 * (1.0 - t) * t * xc + (t ** 2) * x1
            by = ((1.0 - t) ** 2) * y0 + 2.0 * (1.0 - t) * t * yc + (t ** 2) * y1
            curve_x.append(bx)
            curve_y.append(by)

        hover_info = (
            f"<b>{u} ➔ {v}</b><br>"
            f"Conversion Rate: {rate:.6f}<br>"
            f"Log Weight: {weight:.6f}<br>"
            f"{'★ PART OF ARBITRAGE CYCLE' if is_cycle_edge else ''}"
        )

        fig.add_trace(
            go.Scatter(
                x=curve_x,
                y=curve_y,
                mode="lines",
                line=dict(color=edge_color, width=edge_width),
                hoverinfo="text",
                text=hover_info,
                showlegend=False,
            )
        )

        # Draw directed arrowhead near target vertex (at t ≈ 0.88)
        arrow_t = 0.88
        ax_pt = ((1.0 - arrow_t) ** 2) * x0 + 2.0 * (1.0 - arrow_t) * arrow_t * xc + (arrow_t ** 2) * x1
        ay_pt = ((1.0 - arrow_t) ** 2) * y0 + 2.0 * (1.0 - arrow_t) * arrow_t * yc + (arrow_t ** 2) * y1

        prev_t = 0.80
        prev_x = ((1.0 - prev_t) ** 2) * x0 + 2.0 * (1.0 - prev_t) * prev_t * xc + (prev_t ** 2) * x1
        prev_y = ((1.0 - prev_t) ** 2) * y0 + 2.0 * (1.0 - prev_t) * prev_t * yc + (prev_t ** 2) * y1

        annotations.append(
            dict(
                x=ax_pt,
                y=ay_pt,
                ax=prev_x,
                ay=prev_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5 if is_cycle_edge else 1.0,
                arrowwidth=2.8 if is_cycle_edge else 1.4,
                arrowcolor=edge_color,
            )
        )

        # Rate label badge near apex of curve (t ≈ 0.50)
        rate_str = f"{rate:.4g}"
        annotations.append(
            dict(
                x=xc,
                y=yc,
                text=rate_str,
                showarrow=False,
                font=dict(
                    size=10,
                    color="#34D399" if is_cycle_edge else "#CBD5E1",
                    family="ui-monospace, Consolas, monospace",
                ),
                bgcolor="#0F172A",
                bordercolor=edge_color if is_cycle_edge else "#26364C",
                borderwidth=1.5 if is_cycle_edge else 1.0,
                borderpad=3,
            )
        )

    # Draw vertices
    node_x = [coords[n][0] for n in nodes]
    node_y = [coords[n][1] for n in nodes]

    cycle_nodes_set = set(highlight_cycle) if highlight_cycle else set()
    node_colors = [
        "#065F46" if n in cycle_nodes_set else "#1E3A8A"
        for n in nodes
    ]
    node_borders = [
        "#34D399" if n in cycle_nodes_set else "#06B6D4"
        for n in nodes
    ]
    border_widths = [
        3.5 if n in cycle_nodes_set else 2.5
        for n in nodes
    ]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=46,
                color=node_colors,
                line=dict(color=node_borders, width=border_widths),
                opacity=0.98,
            ),
            text=nodes,
            textposition="middle center",
            textfont=dict(size=12, color="#F8FAFC", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
            hoverinfo="text",
            hovertext=[f"Currency Vertex: <b>{n}</b>" for n in nodes],
            showlegend=False,
        )
    )

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=540,
        margin=dict(l=30, r=30, t=30, b=30),
        plot_bgcolor="#070B16",
        paper_bgcolor="#070B16",
    )

    return fig


# ==============================================================================
# Main Streamlit Application
# ==============================================================================

def main() -> None:
    st.set_page_config(
        page_title="CryptoLab — Arbitrage Detection & Analysis",
        page_icon="🪙",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --------------------------------------------------------------------------
    # Custom CSS Injection: Midnight Navy + Electric Cyan Theme
    # --------------------------------------------------------------------------
    custom_css = """
    <style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #070B16;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    }

    /* Sidebar Theme */
    [data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid #26364C;
    }
    [data-testid="stSidebar"] hr {
        border-color: #26364C;
        margin: 1.25rem 0;
    }

    /* Tab bar styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0B1120;
        padding: 6px 8px;
        border-radius: 10px;
        border: 1px solid #26364C;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 8px 18px;
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC;
        background-color: rgba(6, 182, 212, 0.08);
    }
    .stTabs [aria-selected="true"] {
        color: #06B6D4 !important;
        background-color: #0F172A !important;
        border-bottom: 2px solid #06B6D4 !important;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.2);
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(7, 11, 22, 0.98) 100%);
        border: 1px solid #26364C;
        border-radius: 14px;
        padding: 1.5rem 1.85rem;
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #06B6D4, #2563EB, #34D399);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(6, 182, 212, 0.12);
        color: #06B6D4;
        border: 1px solid rgba(6, 182, 212, 0.3);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
        line-height: 1.2;
    }
    .hero-title span {
        color: #06B6D4;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #94A3B8;
        font-style: italic;
        margin-top: 0.25rem;
        margin-bottom: 0.65rem;
    }
    .hero-desc {
        font-size: 0.92rem;
        color: #CBD5E1;
        line-height: 1.55;
        max-width: 950px;
        margin: 0;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #26364C;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #06B6D4;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.25rem;
    }
    .metric-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .metric-val.cyan { color: #06B6D4; }
    .metric-val.green { color: #34D399; }
    .metric-val.amber { color: #F59E0B; }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Content Cards */
    .crypto-panel {
        background-color: #0F172A;
        border: 1px solid #26364C;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    }

    /* Path Pill Badges */
    .path-box {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        background: #0B1120;
        border: 1px solid #26364C;
        border-radius: 10px;
        padding: 12px 18px;
        margin: 12px 0;
    }
    .curr-badge {
        background: rgba(37, 99, 235, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(37, 99, 235, 0.5);
        padding: 4px 12px;
        border-radius: 6px;
        font-family: ui-monospace, Consolas, monospace;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .curr-badge.active {
        background: rgba(52, 211, 153, 0.18);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.6);
    }
    .path-arrow {
        color: #06B6D4;
        font-weight: bold;
        font-size: 1.1rem;
    }

    /* Callout Card */
    .custom-callout {
        background: #0F172A;
        border-left: 4px solid #06B6D4;
        border-top: 1px solid #26364C;
        border-right: 1px solid #26364C;
        border-bottom: 1px solid #26364C;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #CBD5E1;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .custom-callout.warning {
        border-left-color: #F59E0B;
        background: rgba(245, 158, 11, 0.05);
    }
    .custom-callout.success {
        border-left-color: #34D399;
        background: rgba(52, 211, 153, 0.05);
    }
    .custom-callout.danger {
        border-left-color: #EF4444;
        background: rgba(239, 68, 68, 0.05);
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-pill.green {
        background: rgba(52, 211, 153, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
    }
    .badge-pill.slate {
        background: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }
    .badge-pill.cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #06B6D4;
        border: 1px solid rgba(6, 182, 212, 0.4);
    }

    /* Table row hover */
    .table-row-hover:hover {
        background-color: rgba(6, 182, 212, 0.08) !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Session State Initialization
    # --------------------------------------------------------------------------
    if "rates" not in st.session_state:
        st.session_state["rates"] = get_arbitrage_sample_rates()

    if "current_preset" not in st.session_state:
        st.session_state["current_preset"] = "arbitrage"

    # --------------------------------------------------------------------------
    # Header & Project Overview (Requirement 4.A)
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">🔬 Discrete Mathematics Research Laboratory • Graph Theory</div>
            <h1 class="hero-title">🪙 CryptoLab <span>—</span> Cryptocurrency Arbitrage Detection</h1>
            <div class="hero-subtitle">Cryptocurrency Arbitrage Detection & Mathematical Analysis</div>
            <p class="hero-desc">
                An interactive computer engineering laboratory modeling cryptocurrency exchange markets as directed
                weighted graphs. Applies logarithmic reductions and the custom <strong>Bellman–Ford negative-weight cycle
                detection algorithm</strong> to detect theoretical triangular and polygonal arbitrage opportunities in
                simulated exchange-rate networks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prominent simulation disclaimer banner
    st.markdown(
        f"""
        <div class="custom-callout warning">
            ⚠️ <strong>{SIMULATION_DISCLAIMER}</strong> This research dashboard demonstrates theoretical graph modeling
            and algorithmic cycle detection using synthetic data. It does not retrieve live market price feeds, connect
            to cryptocurrency wallets, or execute actual financial trades.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Theoretical Foundations & Graph Transformation Methodology", expanded=False):
        st.markdown(
            r"""
            ### What is Cryptocurrency Arbitrage?
            **Arbitrage** is the simultaneous conversion of capital across multiple currency pairs
            to exploit pricing discrepancies and yield a risk-free gain. In foreign exchange
            and cryptocurrency markets, this often occurs across three or more assets as **triangular**
            or **polygonal arbitrage**.

            ### The Graph Theory Transformation
            1. **Directed Currency Digraph ($G = (V, E)$):** Each cryptocurrency is modeled as a vertex $v \in V$,
               and each executable conversion is modeled as a directed edge $(u, v) \in E$ with conversion rate $R_{u,v} > 0$.
            2. **Multiplicative Arbitrage Condition:** A closed directed cycle $C = (c_0, c_1, \dots, c_k = c_0)$
               is profitable if and only if the product of its conversion rates exceeds unity:
               $$\prod_{i=1}^k R_i > 1.0$$
            3. **Logarithmic Edge Weights:** Standard shortest-path algorithms require additive edge weights.
               Taking the negative natural logarithm of both sides transforms multiplication into addition:
               $$\prod_{i=1}^k R_i > 1.0 \iff \sum_{i=1}^k -\ln(R_i) < 0.0$$
            4. **Equivalence Theorem:** A profitable arbitrage cycle exists **if and only if** the directed graph
               contains a **negative-weight cycle** with edge weights $w(u, v) = -\ln(R_{u,v})$.
            5. **Bellman-Ford Algorithm:** We execute a custom Bellman-Ford detector in $\mathcal{O}(|V| \cdot |E|)$ time
               with multi-component zero-initialization to identify and reconstruct profitable cycles.
            """
        )

    # --------------------------------------------------------------------------
    # Sidebar: Control Panel (Requirement 4.C)
    # --------------------------------------------------------------------------
    st.sidebar.markdown(
        """
        <div style="margin-bottom: 14px;">
            <span class="badge-pill cyan">Control Panel</span>
            <h2 style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin-top: 6px; margin-bottom: 0;">
                ⚙️ Simulation Configuration
            </h2>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
                Select simulated scenarios or specify custom conversion rates.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### 📊 Exchange-Rate Scenario")
    preset_options = [
        "Triangular Arbitrage (+2.5% Gross)",
        "Indian Market INR Arbitrage (+2.27% Gross)",
        "Fee-Consumed Opportunity (+0.5% Gross)",
        "Market Equilibrium (No Arbitrage)",
    ]

    current_idx = 0
    if st.session_state["current_preset"] == "inr_arbitrage":
        current_idx = 1
    elif st.session_state["current_preset"] == "fee_consumed":
        current_idx = 2
    elif st.session_state["current_preset"] == "equilibrium":
        current_idx = 3

    preset_choice = st.sidebar.radio(
        "Load Preset Market Scenario:",
        options=preset_options,
        index=current_idx,
    )

    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("📥 Apply Preset", use_container_width=True):
            if "Indian" in preset_choice or "INR" in preset_choice:
                st.session_state["rates"] = get_inr_arbitrage_sample_rates()
                st.session_state["current_preset"] = "inr_arbitrage"
                st.sidebar.success("Loaded Indian Market INR arbitrage dataset (+2.27%).")
            elif "Triangular" in preset_choice:
                st.session_state["rates"] = get_arbitrage_sample_rates()
                st.session_state["current_preset"] = "arbitrage"
                st.sidebar.success("Loaded triangular arbitrage dataset (+2.5%).")
            elif "Fee-Consumed" in preset_choice:
                st.session_state["rates"] = get_fee_consumed_sample_rates()
                st.session_state["current_preset"] = "fee_consumed"
                st.sidebar.success("Loaded fee-consumed dataset (+0.5%).")
            else:
                st.session_state["rates"] = get_sample_rates()
                st.session_state["current_preset"] = "equilibrium"
                st.sidebar.success("Loaded equilibrium dataset.")
            st.rerun()

    with col_btn2:
        if st.button("🔄 Reset Default", use_container_width=True):
            st.session_state["rates"] = get_arbitrage_sample_rates()
            st.session_state["current_preset"] = "arbitrage"
            st.sidebar.info("Reset to default rates.")
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ➕ Custom Conversion Rates")

    with st.sidebar.form("add_rate_form", clear_on_submit=False):
        new_source = st.text_input("Source Currency (e.g. BTC):", value="").strip().upper()
        new_target = st.text_input("Target Currency (e.g. ETH):", value="").strip().upper()
        new_rate_str = st.text_input("Exchange Rate (units of target per source):", value="").strip()
        submit_rate = st.form_submit_button("➕ Save Conversion Rate", use_container_width=True)

        if submit_rate:
            if not new_source or not new_target:
                st.error("Both Source and Target currency symbols are required.")
            elif new_source == new_target:
                st.error("Self-conversion is not permitted (Source and Target must differ).")
            else:
                try:
                    parsed_rate = float(new_rate_str)
                    if parsed_rate <= 0 or math.isnan(parsed_rate) or math.isinf(parsed_rate):
                        st.error("Exchange rate must be a finite, strictly positive number (> 0).")
                    else:
                        validated_rate = validate_conversion_rate(
                            new_source, new_target, parsed_rate
                        )
                        # Replace existing edge or append new
                        updated_rates = [
                            r for r in st.session_state["rates"]
                            if not (r.source == validated_rate.source and r.target == validated_rate.target)
                        ]
                        updated_rates.append(validated_rate)
                        st.session_state["rates"] = updated_rates
                        st.session_state["current_preset"] = "custom"
                        st.success(f"Added rate: 1 {new_source} = {parsed_rate} {new_target}")
                        st.rerun()
                except (ValueError, TypeError) as err:
                    st.error(f"Invalid rate: {err}")
                except Exception as err:
                    st.error(f"Error saving conversion rate: {err}")

    # Build graph from current session state rates
    current_rates: List[ConversionRate] = st.session_state["rates"]
    graph = build_graph(current_rates)

    # Run Bellman-Ford arbitrage detector
    arb_result: ArbitrageResult = detect_arbitrage(graph)

    # --------------------------------------------------------------------------
    # Dashboard Summary Cards (Requirement 4.B)
    # --------------------------------------------------------------------------
    col_sc1, col_sc2, col_sc3, col_sc4 = st.columns(4)

    with col_sc1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Currency Vertices (|V|)</div>
                <div class="metric-val cyan">{graph.number_of_nodes()}</div>
                <div class="metric-sub">Distinct currency assets in network</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_sc2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Directed Edges (|E|)</div>
                <div class="metric-val">{graph.number_of_edges()}</div>
                <div class="metric-sub">Executable conversion paths</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_sc3:
        status_val_class = "green" if arb_result.has_arbitrage else "slate"
        status_text = "Opportunity Detected" if arb_result.has_arbitrage else "Equilibrium"
        status_badge_class = "green" if arb_result.has_arbitrage else "slate"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Arbitrage Status</div>
                <div class="metric-val {status_val_class}" style="font-size: 1.15rem; font-family: -apple-system, sans-serif;">
                    <span class="badge-pill {status_badge_class}">{status_text}</span>
                </div>
                <div class="metric-sub">{"Negative cycle detected" if arb_result.has_arbitrage else "No negative cycles found"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_sc4:
        if arb_result.has_arbitrage and arb_result.rate_product:
            gross_gain = (arb_result.rate_product - 1.0) * 100.0
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Theoretical Gross Yield</div>
                    <div class="metric-val green">+{gross_gain:.2f}%</div>
                    <div class="metric-sub">Rate Product: {arb_result.rate_product:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Theoretical Gross Yield</div>
                    <div class="metric-val" style="color: #94A3B8;">0.00%</div>
                    <div class="metric-sub">Rate Product: 1.0000</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------------------------
    # Main Dashboard Tabs (Requirement 4.D)
    # --------------------------------------------------------------------------
    tab_overview, tab_graph, tab_detector, tab_profit, tab_math = st.tabs(
        [
            "📊 Overview & Rates",
            "🕸️ Currency Network",
            "🔍 Arbitrage Detection",
            "💰 Profit & Fee Simulator",
            "📐 Mathematical Framework",
        ]
    )

    # ==========================================================================
    # Tab 1: Exchange Rates Table
    # ==========================================================================
    with tab_overview:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                <h3 style="margin: 0; color: #F8FAFC; font-weight: 700;">Directed Exchange Rates Matrix</h3>
                <span class="badge-pill cyan">Weight Mapping: w(u, v) = -ln(Rate)</span>
            </div>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 12px;">
                Each row corresponds to a directed conversion edge in digraph <em>G</em>. A negative logarithmic weight
                indicates an edge whose reciprocal is favorable to arbitrage candidates.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Build table rows
        rate_headers = ["Source Currency", "Direction", "Target Currency", "Exchange Rate", "Log Weight: -ln(Rate)"]
        rate_rows = []
        for r in current_rates:
            w = calculate_log_weight(r.rate)
            rate_rows.append([
                r.source,
                "➔",
                r.target,
                f"{r.rate:.6f}",
                f"{w:+.6f}",
            ])

        render_table(rate_headers, rate_rows)

    # ==========================================================================
    # Tab 2: Directed Graph Visualization (Requirement 4.E)
    # ==========================================================================
    with tab_graph:
        st.markdown(
            """
            <h3 style="margin: 0; color: #F8FAFC; font-weight: 700;">Currency Network Digraph</h3>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 12px;">
                Interactive Plotly topology of currency vertices and directed conversion paths.
                Curved directed arrows depict conversion paths; detected arbitrage cycle edges are illuminated in green.
            </div>
            """,
            unsafe_allow_html=True,
        )

        cycle_to_highlight = arb_result.cycle if arb_result.has_arbitrage else None
        fig = create_directed_network_figure(graph, highlight_cycle=cycle_to_highlight)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div style="display: flex; gap: 16px; flex-wrap: wrap; background-color: #0F172A; border: 1px solid #26364C; border-radius: 8px; padding: 10px 16px; font-size: 0.82rem; color: #94A3B8;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 14px; height: 14px; border-radius: 50%; background-color: #1E3A8A; border: 2px solid #06B6D4;"></span>
                    <span>Standard Currency Vertex</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 14px; height: 14px; border-radius: 50%; background-color: #065F46; border: 2px solid #34D399;"></span>
                    <span>Arbitrage Cycle Vertex</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 20px; height: 3px; background-color: rgba(6, 182, 212, 0.6);"></span>
                    <span>Standard Directed Conversion Edge</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 20px; height: 4px; background-color: #34D399;"></span>
                    <span>Candidate Arbitrage Cycle Path</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==========================================================================
    # Tab 3: Arbitrage Detection Results (Requirement 4.F)
    # ==========================================================================
    with tab_detector:
        st.markdown(
            """
            <h3 style="margin: 0; color: #F8FAFC; font-weight: 700;">Bellman–Ford Negative-Weight Cycle Detection</h3>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 16px;">
                Identifies closed directed walks where edge relaxation succeeds on pass |V|, signaling a negative-weight cycle.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if arb_result.has_arbitrage and arb_result.cycle:
            st.markdown(
                """
                <div class="custom-callout success">
                    🎯 <strong>Arbitrage Opportunity Identified!</strong> A directed negative-weight cycle exists
                    in the simulated currency graph. The product of exchange rates around this closed walk exceeds 1.0.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Styled cycle path badges
            path_pills = []
            for idx, c in enumerate(arb_result.cycle):
                path_pills.append(f'<span class="curr-badge active">{c}</span>')
                if idx < len(arb_result.cycle) - 1:
                    path_pills.append('<span class="path-arrow">➔</span>')

            st.markdown(
                f"""
                <div style="margin-bottom: 16px;">
                    <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #94A3B8; margin-bottom: 6px;">
                        Reconstructed Directed Cycle Sequence:
                    </div>
                    <div class="path-box">
                        {" ".join(path_pills)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_a1, col_a2, col_a3, col_a4 = st.columns(4)
            with col_a1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Gross Rate Product (∏ R)</div>
                        <div class="metric-val cyan">{arb_result.rate_product:.6f}</div>
                        <div class="metric-sub">> 1.0000 indicates gross profit</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_a2:
                gross_gain = (arb_result.rate_product - 1.0) * 100.0  # type: ignore
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Gross Yield</div>
                        <div class="metric-val green">+{gross_gain:.4f}%</div>
                        <div class="metric-sub">Compounded gross return</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_a3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Log Weight Sum (∑ -ln R)</div>
                        <div class="metric-val green">{arb_result.log_weight_sum:.6f}</div>
                        <div class="metric-sub">< 0 indicates negative cycle</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_a4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Conversion Hops (k)</div>
                        <div class="metric-val">{len(arb_result.cycle) - 1}</div>
                        <div class="metric-sub">Number of sequential trades</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### Directed Cycle Trade Breakdown")
            cycle_headers = ["Step", "From", "➔", "To", "Exchange Rate", "Logarithmic Weight"]
            cycle_rows = []
            for i in range(len(arb_result.cycle) - 1):
                u = arb_result.cycle[i]
                v = arb_result.cycle[i + 1]
                edge_rate = graph[u][v]["rate"]
                edge_weight = graph[u][v]["weight"]
                cycle_rows.append([
                    f"Hop {i + 1}",
                    u,
                    "➔",
                    v,
                    f"{edge_rate:.6f}",
                    f"{edge_weight:+.6f}",
                ])
            render_table(cycle_headers, cycle_rows)

            st.markdown(
                """
                <div class="custom-callout">
                    💡 <strong>Academic Notice: Theoretical Gross Candidate vs. Live Execution Reality</strong><br>
                    The detected negative cycle represents an idealized mathematical arbitrage condition.
                    In live production markets, transaction fees, bid-ask spread, order book depth (slippage),
                    and latency compound across each hop and can rapidly eliminate net profitability.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="custom-callout">
                    ⚖️ <strong>No negative cycle detected in the current simulated exchange-rate network.</strong><br><br>
                    All shortest-path estimates converged within |V| - 1 edge relaxation passes, confirming that no edge
                    can be relaxed further on pass |V|. The simulated exchange rates are in mathematical equilibrium,
                    and no risk-free arbitrage cycle exists.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==========================================================================
    # Tab 4: Profit & Fee Simulator (Requirement 4.G)
    # ==========================================================================
    with tab_profit:
        st.markdown(
            """
            <h3 style="margin: 0; color: #F8FAFC; font-weight: 700;">Arbitrage Profit & Transaction-Fee Simulator</h3>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 16px;">
                Simulates capital compounding across the detected closed cycle, subtracting per-hop exchange fees
                to calculate net returns and determine the breakeven fee threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not arb_result.has_arbitrage or not arb_result.cycle:
            st.markdown(
                """
                <div class="custom-callout warning">
                    ⚠️ <strong>Simulation Unavailable:</strong> No closed arbitrage cycle was detected in the current currency graph.<br>
                    Please select the <strong>'Triangular Arbitrage'</strong> or <strong>'Indian Market INR Arbitrage'</strong>
                    preset in the sidebar to simulate compounding yields.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                initial_capital = st.number_input(
                    f"Initial Investment Capital ({arb_result.cycle[0]}):",
                    min_value=0.01,
                    max_value=1_000_000.0,
                    value=1000.0,
                    step=100.0,
                )
            with col_p2:
                default_fee = 0.20 if st.session_state["current_preset"] == "fee_consumed" else 0.10
                trade_fee_percent = st.slider(
                    "Transaction Fee per Conversion Hop (%):",
                    min_value=0.0,
                    max_value=3.0,
                    value=default_fee,
                    step=0.01,
                    format="%.2f%%",
                )

            # Execute profit calculator
            profit_res: ProfitResult = calculate_profit(
                graph=graph,
                cycle=arb_result.cycle,
                initial_investment=initial_capital,
                fee_percent=trade_fee_percent,
            )

            # Breakeven fee calculation: (1 - f*)^k = 1 / Product
            k_hops = profit_res.num_conversions
            breakeven_fee = (1.0 - (1.0 / arb_result.rate_product) ** (1.0 / k_hops)) * 100.0  # type: ignore

            # Summary Metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Initial Capital</div>
                        <div class="metric-val">{profit_res.initial_investment:,.2f}</div>
                        <div class="metric-sub">{profit_res.start_currency} Principal</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Gross Final Amount</div>
                        <div class="metric-val cyan">{profit_res.gross_final_amount:,.2f}</div>
                        <div class="metric-sub">{profit_res.gross_return_percent:+.2f}% Gross Return</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col3:
                net_color = "green" if profit_res.is_profitable else "amber"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Net Final (After Fees)</div>
                        <div class="metric-val {net_color}">{profit_res.net_final_amount:,.2f}</div>
                        <div class="metric-sub">{profit_res.net_return_percent:+.4f}% Net Return</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Breakeven Fee Threshold (f*)</div>
                        <div class="metric-val cyan">{breakeven_fee:.4f}%</div>
                        <div class="metric-sub">Maximum allowable fee / hop</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if profit_res.is_profitable:
                st.markdown(
                    f"""
                    <div class="custom-callout success">
                        ✅ <strong>Net Profitable Arbitrage:</strong> After deducting a fee of {trade_fee_percent:.2f}%
                        across {k_hops} conversion hops, net profit is
                        <strong>{profit_res.net_profit:+.4f} {profit_res.start_currency}</strong>
                        ({profit_res.net_return_percent:+.4f}% net return).
                        The fee rate of {trade_fee_percent:.2f}% is safely beneath the breakeven threshold of {breakeven_fee:.4f}%.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="custom-callout danger">
                        ❌ <strong>Fee-Consumed Opportunity (Net Loss):</strong> The simulated transaction fee
                        ({trade_fee_percent:.2f}% per hop) exceeds the breakeven threshold of {breakeven_fee:.4f}%.
                        Cumulative trading fees eliminate the gross discrepancy, resulting in a net loss of
                        <strong>{profit_res.net_profit:+.4f} {profit_res.start_currency}</strong>
                        ({profit_res.net_return_percent:+.4f}% net loss).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### Step-by-Step Multi-Hop Conversion Audit")
            step_headers = [
                "Step",
                "Conversion",
                "Input Amount",
                "Rate",
                "Gross Received",
                f"Fee ({trade_fee_percent:.2f}%)",
                "Net Retained Amount",
            ]
            step_rows = []
            for s in profit_res.steps:
                step_rows.append([
                    f"Hop {s.step_number}",
                    f"{s.from_currency} ➔ {s.to_currency}",
                    f"{s.amount_in:,.4f} {s.from_currency}",
                    f"{s.rate:.6f}",
                    f"{s.gross_amount_out:,.4f} {s.to_currency}",
                    f"{s.fee_amount:,.4f} {s.to_currency}",
                    f"{s.net_amount_out:,.4f} {s.to_currency}",
                ])
            render_table(step_headers, step_rows)

            st.markdown(
                """
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 8px;">
                    🔬 <strong>Simulation Model Limitations:</strong> Assumes atomic, full fills at quoted exchange rates with zero slippage.
                    In live environments, blockchain gas fees, order book depth limits, and exchange withdrawal costs also apply.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==========================================================================
    # Tab 5: Mathematical Framework (Requirement 4.H)
    # ==========================================================================
    with tab_math:
        st.markdown(
            """
            <h3 style="margin: 0; color: #F8FAFC; font-weight: 700;">Mathematical Framework & Discrete Foundations</h3>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 20px;">
                Theoretical reduction of cryptocurrency exchange rate discrepancies to negative-weight cycles in directed graphs.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""
            ### 1. Directed Weighted Graphs ($G = (V, E)$)
            In discrete mathematics, currency exchange networks are modeled as a **directed graph** $G = (V, E)$:
            - **Vertices ($V$):** Distinct currencies or digital tokens (e.g., BTC, ETH, USDT, SOL).
            - **Edges ($E$):** Directed pairs $(u, v)$ indicating that currency $u$ can be converted directly into currency $v$.
            - **Conversion Rate ($R_{u,v}$):** Multiplicative factor such that $1 \text{ unit of } u = R_{u,v} \text{ units of } v$.

            ---

            ### 2. Multiplicative to Additive Logarithmic Transformation
            A directed cycle $C = (c_0, c_1, \dots, c_k = c_0)$ represents an arbitrage opportunity if:
            $$\prod_{i=1}^k R_{c_{i-1}, c_i} > 1.0$$

            Standard shortest path algorithms (Dijkstra, Bellman-Ford) compute path lengths by **adding** edge weights,
            not multiplying them. By taking the natural logarithm ($\ln$):
            $$\ln\left(\prod_{i=1}^k R_i\right) > \ln(1.0) = 0 \iff \sum_{i=1}^k \ln(R_i) > 0$$

            Multiplying the entire inequality by $-1$ reverses the inequality:
            $$\sum_{i=1}^k \left(-\ln(R_i)\right) < 0.0$$

            Therefore, by defining edge weights as:
            $$w(u, v) = -\ln(R_{u,v})$$
            every profitable arbitrage cycle maps **isomorphically** to a **negative-weight cycle** in the digraph.

            ---

            ### 3. Bellman–Ford Algorithm & Negative Cycle Detection
            """
        )

        st.markdown(
            """
            <div style="display: flex; gap: 12px; margin-bottom: 16px;">
                <span class="badge-pill cyan">Time Complexity: O(|V| × |E|)</span>
                <span class="badge-pill slate">Space Complexity: O(|V|)</span>
                <span class="badge-pill green">Negative Cycle Detection: Pass |V|</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""
            The Bellman–Ford algorithm solves the shortest paths problem in $\mathcal{O}(|V| \cdot |E|)$ time:
            1. **Multi-Component Distance Initialization:**
               $$d[v] = 0.0 \quad \forall v \in V$$
               Initializing all distances to $0.0$ conceptually creates a virtual source connected to every vertex with zero-weight edges,
               guaranteeing detection across disconnected subgraphs without arbitrary source selection.
            2. **Relaxation Loop ($|V| - 1$ passes):**
               For each directed edge $(u, v) \in E$:
               $$\text{if } d[u] + w(u, v) < d[v]: \quad d[v] = d[u] + w(u, v), \quad \pi[v] = u$$
            3. **Negative Cycle Verification (Pass $|V|$):**
               In a cycle-free graph, all shortest paths contain at most $|V| - 1$ edges. If any edge can *still* relax on pass $|V|$,
               a negative-weight cycle is guaranteed to exist by the **Pigeonhole Principle**.
            4. **Predecessor Backtracking:**
               To extract the exact cycle, backtrack $|V|$ times along predecessor pointers $\pi$ to enter the cycle,
               then trace backwards until the cycle repeats, and reverse the sequence to obtain the forward trade path.

            ---

            ### 4. Fee-Adjusted Compounding Model & Breakeven Threshold
            At each conversion hop $i$, a transaction fee percentage $f \in [0, 100)\%$ is deducted from the received currency:
            $$A_i = A_{i-1} \times R_i \times \left(1 - \frac{f}{100}\right)$$

            Compounding across all $k$ hops:
            $$A_{\text{net}} = A_0 \times \left(\prod_{i=1}^k R_i\right) \times \left(1 - \frac{f}{100}\right)^k$$

            **Breakeven Fee Threshold ($f^*$):**
            Setting $A_{\text{net}} = A_0$ yields:
            $$f^* = \left[1 - \left(\frac{1}{\prod_{i=1}^k R_i}\right)^{1/k}\right] \times 100\%$$

            ---

            ### 5. Model Assumptions and Real-World Limitations
            The simulation model in CryptoLab is an idealized educational laboratory. Production trading systems encounter:
            1. **Order Book Depth & Slippage:** Orders are assumed to fill entirely at the top-of-book quoted spot rate; large orders in real markets walk the order book, worsening average execution prices.
            2. **Liquidity Constraints:** Quoted rates assume infinite liquidity. Real markets have finite volume per price level.
            3. **Withdrawal Charges:** Transferring funds between centralized exchanges incurs flat or percentage withdrawal fees.
            4. **Blockchain Gas Fees:** On-chain decentralized exchange (DEX) swaps incur blockchain transaction fees that are independent of trade volume.
            5. **Execution Latency:** High-frequency trading (HFT) bots and miner-extractable value (MEV) searchers detect and close discrepancies within milliseconds.
            """
        )


if __name__ == "__main__":
    main()
