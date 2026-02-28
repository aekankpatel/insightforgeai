"""
InsightForge AI - Visualization Engine
Generates Plotly charts for EDA results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


# ── Color palette ──────────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#00D4FF",
    "secondary": "#7B61FF",
    "accent": "#FF6B6B",
    "success": "#00E5A0",
    "warning": "#FFB800",
    "bg": "#0A0E1A",
    "surface": "#111827",
    "border": "#1F2937",
    "text": "#E2E8F0",
    "muted": "#64748B",
}

CHART_LAYOUT = dict(
    paper_bgcolor=PALETTE["surface"],
    plot_bgcolor=PALETTE["surface"],
    font=dict(color=PALETTE["text"], family="'DM Mono', monospace"),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
              PALETTE["success"], PALETTE["warning"]],
)


def missing_heatmap(profile: dict) -> go.Figure:
    """Horizontal bar chart of missing value percentages."""
    missing_data = {
        col: info["pct"]
        for col, info in profile["missing"].items()
        if info["count"] > 0
    }

    if not missing_data:
        fig = go.Figure()
        fig.add_annotation(text="✓ No missing values detected", x=0.5, y=0.5,
                           font=dict(size=16, color=PALETTE["success"]),
                           showarrow=False, xref="paper", yref="paper")
        fig.update_layout(**CHART_LAYOUT, title="Missing Values", height=200)
        return fig

    cols = list(missing_data.keys())
    pcts = list(missing_data.values())
    colors = [PALETTE["accent"] if p > 20 else PALETTE["warning"] if p > 5 else PALETTE["primary"]
              for p in pcts]

    fig = go.Figure(go.Bar(
        x=pcts, y=cols, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Missing: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text="Missing Values (%)", font=dict(size=14)),
                      xaxis=dict(title="% Missing", gridcolor=PALETTE["border"]),
                      yaxis=dict(gridcolor=PALETTE["border"]),
                      height=max(200, len(cols) * 40 + 80))
    return fig


def distribution_plots(df: pd.DataFrame, max_cols: int = 6) -> go.Figure:
    """Grid of histograms for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if not numeric_cols:
        return None

    n = len(numeric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=numeric_cols,
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    for i, col in enumerate(numeric_cols):
        row, col_idx = divmod(i, ncols)
        data = df[col].dropna()

        fig.add_trace(
            go.Histogram(
                x=data, name=col,
                marker=dict(color=PALETTE["primary"], opacity=0.8,
                            line=dict(color=PALETTE["bg"], width=0.5)),
                nbinsx=30,
                hovertemplate=f"<b>{col}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ),
            row=row + 1, col=col_idx + 1,
        )

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Feature Distributions", font=dict(size=14)),
        showlegend=False,
        height=280 * nrows,
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor=PALETTE["border"], showgrid=True)

    return fig


def correlation_heatmap(correlation: dict) -> go.Figure:
    """Interactive correlation matrix heatmap."""
    matrix = correlation.get("matrix", {})
    if not matrix:
        return None

    cols = list(matrix.keys())
    z = [[matrix[r][c] for c in cols] for r in cols]

    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=cols,
        colorscale=[
            [0, "#FF6B6B"], [0.5, PALETTE["surface"]], [1, PALETTE["primary"]]
        ],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{matrix[r][c]:.2f}" for c in cols] for r in cols],
        texttemplate="%{text}",
        textfont=dict(size=10, color=PALETTE["text"]),
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(
            title="r", tickfont=dict(color=PALETTE["text"]),
            bgcolor=PALETTE["surface"],
        ),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Pearson Correlation Matrix", font=dict(size=14)),
        height=max(350, len(cols) * 50 + 100),
        xaxis=dict(tickangle=-45),
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex color to rgba string for Plotly compatibility."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def outlier_boxplots(df: pd.DataFrame, anomalies: dict, max_cols: int = 9) -> go.Figure:
    """Box plots in a grid — each column on its own subplot to avoid scale distortion."""
    cols_with_outliers = [c for c, v in anomalies.items() if v["iqr_outlier_count"] > 0]
    cols_to_plot = (cols_with_outliers or list(anomalies.keys()))[:max_cols]

    if not cols_to_plot:
        return None

    n = len(cols_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    palette = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
               PALETTE["success"], PALETTE["warning"], "#FF61D8"]

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=cols_to_plot,
                        vertical_spacing=0.1, horizontal_spacing=0.08)

    for i, col in enumerate(cols_to_plot):
        row, col_idx = divmod(i, ncols)
        data = df[col].dropna()
        color = palette[i % len(palette)]
        fig.add_trace(go.Box(
            y=data, name=col,
            marker=dict(color=color, outliercolor=PALETTE["accent"],
                        size=3, line=dict(color=PALETTE["bg"], width=0.5)),
            line=dict(color=color),
            fillcolor=_hex_to_rgba(color, 0.15),
            hovertemplate=f"<b>{col}</b><br>%{{y}}<extra></extra>",
            showlegend=False,
            boxmean=True,
        ), row=row + 1, col=col_idx + 1)

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Outlier Distribution — each column on its own scale", font=dict(size=14)),
        showlegend=False,
        height=280 * nrows,
    )
    fig.update_yaxes(gridcolor=PALETTE["border"])
    return fig


def categorical_bar_charts(df: pd.DataFrame, profile: dict, max_cols: int = 4) -> go.Figure:
    """Bar charts for top categorical columns."""
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()[:max_cols]
    if not cat_cols:
        return None

    n = len(cat_cols)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=cat_cols,
                        vertical_spacing=0.15, horizontal_spacing=0.1)

    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["success"]]

    for i, col in enumerate(cat_cols):
        row, col_idx = divmod(i, ncols)
        vc = df[col].value_counts().head(8)
        fig.add_trace(
            go.Bar(
                x=vc.index.astype(str), y=vc.values,
                name=col,
                marker=dict(color=colors[i % len(colors)], opacity=0.85,
                            line=dict(color=PALETTE["bg"], width=0.5)),
                hovertemplate=f"<b>%{{x}}</b><br>Count: %{{y}}<extra></extra>",
            ),
            row=row + 1, col=col_idx + 1,
        )

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Categorical Feature Distributions", font=dict(size=14)),
        showlegend=False,
        height=320 * nrows,
    )
    return fig


def feature_overview_sunburst(feature_summary: dict) -> go.Figure:
    """Sunburst chart showing feature type breakdown."""
    total_features = sum(len(v) for v in feature_summary.values())
    labels, parents, values, colors_list = ["Features"], [""], [total_features], [PALETTE["surface"]]
    type_colors = {
        "numeric_features": PALETTE["primary"],
        "categorical_features": PALETTE["secondary"],
        "binary_features": PALETTE["success"],
        "high_cardinality_features": PALETTE["warning"],
        "potential_id_columns": PALETTE["accent"],
        "datetime_features": "#FF61D8",
    }
    type_labels = {
        "numeric_features": "Numeric",
        "categorical_features": "Categorical",
        "binary_features": "Binary",
        "high_cardinality_features": "High Cardinality",
        "potential_id_columns": "Potential IDs",
        "datetime_features": "Datetime",
    }

    for key, cols in feature_summary.items():
        if not cols:
            continue
        label = type_labels.get(key, key)
        labels.append(label)
        parents.append("Features")
        values.append(len(cols))
        colors_list.append(type_colors.get(key, PALETTE["muted"]))

        for col in cols:
            labels.append(col)
            parents.append(label)
            values.append(1)
            colors_list.append(_hex_to_rgba(type_colors.get(key, PALETTE["muted"]), 0.6))

    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        marker=dict(colors=colors_list, line=dict(color=PALETTE["bg"], width=1)),
        branchvalues="remainder",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
        textfont=dict(color=PALETTE["text"]),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Feature Type Breakdown", font=dict(size=14)),
        height=420,
    )
    return fig


def correlation_network(correlation: dict) -> go.Figure:
    """Network graph of feature correlations — nodes are features, edges are correlations."""
    pairs = correlation.get("strong_pairs", [])
    if not pairs:
        return None

    # Collect all nodes
    nodes = list({p["feature_a"] for p in pairs} | {p["feature_b"] for p in pairs})
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}

    # Layout: circular
    angles = [2 * np.pi * i / n for i in range(n)]
    xs = [np.cos(a) for a in angles]
    ys = [np.sin(a) for a in angles]

    fig = go.Figure()

    # Draw edges
    for pair in pairs:
        i, j = node_idx[pair["feature_a"]], node_idx[pair["feature_b"]]
        r = abs(pair["correlation"])
        color = PALETTE["accent"] if pair["correlation"] < 0 else PALETTE["primary"]
        fig.add_trace(go.Scatter(
            x=[xs[i], xs[j], None], y=[ys[i], ys[j], None],
            mode="lines",
            line=dict(width=r * 4, color=color),
            opacity=0.5 + r * 0.4,
            hoverinfo="skip",
            showlegend=False,
        ))

    # Draw nodes
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(size=18, color=PALETTE["secondary"],
                    line=dict(color=PALETTE["primary"], width=2)),
        text=nodes,
        textposition="top center",
        textfont=dict(size=10, color=PALETTE["text"]),
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Correlation Network  (cyan = positive · red = negative)", font=dict(size=14)),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=460,
    )
    return fig


def before_after_distributions(df_before: pd.DataFrame, df_after: pd.DataFrame, max_cols: int = 6) -> go.Figure:
    """Side-by-side histograms comparing before vs after cleaning."""
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if not numeric_cols:
        return None

    n = len(numeric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=numeric_cols,
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    for i, col in enumerate(numeric_cols):
        row, col_idx = divmod(i, ncols)
        before_data = df_before[col].dropna()
        after_data  = df_after[col].dropna()

        fig.add_trace(go.Histogram(
            x=before_data, name="Before", nbinsx=25,
            marker=dict(color=_hex_to_rgba(PALETTE["accent"], 0.7), line=dict(width=0)),
            showlegend=(i == 0),
        ), row=row+1, col=col_idx+1)

        fig.add_trace(go.Histogram(
            x=after_data, name="After", nbinsx=25,
            marker=dict(color=_hex_to_rgba(PALETTE["success"], 0.7), line=dict(width=0)),
            showlegend=(i == 0),
        ), row=row+1, col=col_idx+1)

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Before vs After Cleaning", font=dict(size=14)),
        barmode="overlay",
        height=280 * nrows,
        legend=dict(orientation="h", y=1.02, font=dict(color=PALETTE["text"])),
    )
    return fig


def column_distribution_detail(df: pd.DataFrame, col: str) -> go.Figure:
    """Detailed distribution chart for a single column."""
    series = df[col].dropna()

    if pd.api.types.is_numeric_dtype(series):
        # Numeric: histogram + box plot side by side
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Box Plot"])
        fig.add_trace(go.Histogram(
            x=series, nbinsx=30,
            marker=dict(color=PALETTE["primary"], opacity=0.8,
                        line=dict(color=PALETTE["bg"], width=0.5)),
            hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Box(
            y=series, name=col,
            marker=dict(color=PALETTE["primary"], outliercolor=PALETTE["accent"],
                        size=4, line=dict(color=PALETTE["bg"])),
            line=dict(color=PALETTE["primary"]),
            fillcolor=_hex_to_rgba(PALETTE["primary"], 0.2),
            boxmean=True,
        ), row=1, col=2)
        fig.update_layout(
            **CHART_LAYOUT,
            title=dict(text=f"Column: {col}", font=dict(size=14)),
            height=360, showlegend=False,
        )
        fig.update_xaxes(gridcolor=PALETTE["border"])
        fig.update_yaxes(gridcolor=PALETTE["border"])
    else:
        # Categorical: horizontal bar chart (top 15 values)
        vc = series.value_counts().head(15).sort_values()
        bar_colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
                      PALETTE["success"], PALETTE["warning"], "#FF61D8"]
        colors_list = [bar_colors[i % len(bar_colors)] for i in range(len(vc))]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vc.values, y=vc.index.astype(str),
            orientation="h",
            marker=dict(color=colors_list, opacity=0.85),
            hovertemplate="%{y}: %{x}<extra></extra>",
            text=vc.values,
            textposition="outside",
            textfont=dict(color=PALETTE["text"], size=11),
        ))
        fig.update_layout(
            **CHART_LAYOUT,
            title=dict(text=f"Column: {col}  (top {len(vc)} values)", font=dict(size=14)),
            xaxis=dict(gridcolor=PALETTE["border"], title="Count"),
            yaxis=dict(gridcolor=PALETTE["border"]),
            height=max(320, len(vc) * 32 + 80),
            showlegend=False,
        )

    return fig


def health_scorecard_chart(scores: dict) -> go.Figure:
    """Radial bar / gauge chart for dataset health dimensions."""
    dims   = list(scores.keys())
    values = [scores[d]["score"] for d in dims]
    colors = [PALETTE["success"] if v >= 80 else PALETTE["warning"] if v >= 60 else PALETTE["accent"]
              for v in values]

    fig = go.Figure()
    for i, (dim, val, color) in enumerate(zip(dims, values, colors)):
        fig.add_trace(go.Bar(
            x=[val], y=[dim], orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            text=[f"{val}/100"],
            textposition="outside",
            textfont=dict(color=PALETTE["text"], size=12),
            hovertemplate=f"<b>{dim}</b><br>Score: {val}/100<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Dataset Health Scorecard", font=dict(size=14)),
        xaxis=dict(range=[0, 115], gridcolor=PALETTE["border"], title="Score"),
        yaxis=dict(gridcolor=PALETTE["border"]),
        height=300,
        bargap=0.3,
    )
    return fig
