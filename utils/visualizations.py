"""Plotly dark-theme visualizations for statistical tests."""
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# ── shared dark theme ─────────────────────────────────────────────────────────
_BG      = "#161b22"
_BORDER  = "#30363d"
_TEXT    = "#e6edf3"
_MUTED   = "#8b949e"
_BLUE    = "#58a6ff"
_GREEN   = "#3fb950"
_RED     = "#f85149"
_ORANGE  = "#f0883e"
_PURPLE  = "#a371f7"
_PALETTE = [_BLUE, _GREEN, _ORANGE, _PURPLE, _RED]

_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=_BG, plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="sans-serif"),
    margin=dict(l=20, r=20, t=48, b=20),
    hoverlabel=dict(bgcolor="#1c2333", font_color=_TEXT, bordercolor=_BORDER),
)


def _apply(fig, height=420):
    fig.update_layout(**_BASE, height=height)
    fig.update_xaxes(gridcolor=_BORDER, zerolinecolor=_BORDER)
    fig.update_yaxes(gridcolor=_BORDER, zerolinecolor=_BORDER)
    return fig


# ── Distribution + rejection region ──────────────────────────────────────────
def plot_test_distribution(result) -> go.Figure:
    """Show distribution, rejection region, test statistic, and p-value area."""
    stat = result.statistic
    alpha = result.alpha
    tail = result.tail
    dist_type = result.dist_type
    df = result.df

    # x range
    if dist_type == "chi2":
        x_max = max(stats.chi2.ppf(0.999, df), stat * 1.3)
        x = np.linspace(0, x_max, 600)
        y = stats.chi2.pdf(x, df)
        crit_r = stats.chi2.ppf(1 - alpha, df)
        title = f"χ² 分布 (df={df:.1f})"
    elif dist_type == "normal":
        x_lim = max(4.0, abs(stat) * 1.4)
        x = np.linspace(-x_lim, x_lim, 600)
        y = stats.norm.pdf(x)
        crit_r = stats.norm.ppf(1 - alpha/2) if tail == "two" else stats.norm.ppf(1 - alpha)
        crit_l = -crit_r
        title = "標準正規分布 N(0,1)"
    else:  # t
        x_lim = max(4.0, abs(stat) * 1.4)
        x = np.linspace(-x_lim, x_lim, 600)
        y = stats.t.pdf(x, df)
        crit_r = stats.t.ppf(1 - alpha/2, df) if tail == "two" else stats.t.ppf(1 - alpha, df)
        crit_l = -crit_r
        title = f"t 分布 (df={df:.1f})"

    fig = go.Figure()

    # Base distribution
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=_BLUE, width=2),
        name="分布", fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
    ))

    # Rejection region(s)
    def shade_region(x_from, x_to, label="棄却域"):
        mask = (x >= x_from) & (x <= x_to)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=np.concatenate([[x[mask][0]], x[mask], [x[mask][-1]]]),
                y=np.concatenate([[0], y[mask], [0]]),
                fill="toself", mode="none",
                fillcolor="rgba(248,81,73,0.35)",
                name=label, showlegend=True,
            ))

    if dist_type == "chi2":
        shade_region(crit_r, x[-1], "棄却域 (右尾)")
    elif tail == "two":
        shade_region(x[0], crit_l, "棄却域 (左尾)")
        shade_region(crit_r, x[-1], "棄却域 (右尾)")
    elif tail == "left":
        shade_region(x[0], -crit_r, "棄却域 (左尾)")
    else:
        shade_region(crit_r, x[-1], "棄却域 (右尾)")

    # p-value area (between stat and tail)
    def shade_pvalue(x_from, x_to):
        mask = (x >= x_from) & (x <= x_to)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=np.concatenate([[x[mask][0]], x[mask], [x[mask][-1]]]),
                y=np.concatenate([[0], y[mask], [0]]),
                fill="toself", mode="none",
                fillcolor="rgba(240,136,62,0.45)",
                name="p値", showlegend=True,
            ))

    if dist_type != "chi2":
        if tail == "two":
            if stat > 0:
                shade_pvalue(stat, x[-1])
                shade_pvalue(x[0], -stat)
            else:
                shade_pvalue(x[0], stat)
                shade_pvalue(-stat, x[-1])
        elif tail == "left":
            shade_pvalue(x[0], stat)
        else:
            shade_pvalue(stat, x[-1])
    else:
        if stat <= x[-1]:
            shade_pvalue(stat, x[-1])

    # Critical value line(s)
    def vline(xv, label, color):
        fig.add_vline(x=xv, line_dash="dash", line_color=color, line_width=1.5,
                      annotation_text=label, annotation_position="top right",
                      annotation_font_color=color)

    if dist_type == "chi2":
        vline(crit_r, f"臨界値={crit_r:.3f}", _RED)
    elif tail == "two":
        vline(-abs(crit_r), f"−{abs(crit_r):.3f}", _RED)
        vline(abs(crit_r),  f"+{abs(crit_r):.3f}", _RED)
    elif tail == "left":
        vline(-abs(crit_r), f"臨界値={-abs(crit_r):.3f}", _RED)
    else:
        vline(abs(crit_r), f"臨界値={abs(crit_r):.3f}", _RED)

    # Test statistic line
    fig.add_vline(x=stat, line_color=_ORANGE, line_width=2.5,
                  annotation_text=f"検定統計量={stat:.3f}",
                  annotation_position="top left",
                  annotation_font_color=_ORANGE)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="値",
        yaxis_title="確率密度",
        legend=dict(bgcolor=_BG, bordercolor=_BORDER, borderwidth=1, x=0.01, y=0.99),
        showlegend=True,
    )
    return _apply(fig, 440)


# ── Confidence interval ───────────────────────────────────────────────────────
def plot_confidence_interval(result, label1="グループ1", label2="グループ2") -> go.Figure:
    fig = go.Figure()
    ci = result.ci
    if ci is None:
        return fig

    if result.dist_type in ("t", "normal"):
        # For 1-sample or 2-sample: show CI around mean (or difference)
        m = result.mean1 if result.mean2 is None else (result.mean1 - result.mean2)
        label = "μ の推定" if result.mean2 is None else "μ₁−μ₂ の推定"

        # CI bar
        fig.add_trace(go.Scatter(
            x=[ci[0], ci[1]], y=[0, 0], mode="lines",
            line=dict(color=_BLUE, width=4), name=f"{int((1-result.alpha)*100)}% CI",
        ))
        # End caps
        for xv in [ci[0], ci[1]]:
            fig.add_trace(go.Scatter(
                x=[xv, xv], y=[-0.15, 0.15], mode="lines",
                line=dict(color=_BLUE, width=2.5), showlegend=False,
            ))
        # Point estimate
        fig.add_trace(go.Scatter(
            x=[m], y=[0], mode="markers",
            marker=dict(color=_ORANGE, size=12, symbol="circle"),
            name="点推定値",
        ))
        # H0 reference
        h0_val = None
        try:
            h0_val = float(result.h0.split("=")[1].strip().split()[0])
        except Exception:
            pass
        if h0_val is not None:
            fig.add_vline(x=h0_val, line_dash="dash", line_color=_RED, line_width=1.5,
                          annotation_text=f"H₀: {result.h0}",
                          annotation_font_color=_RED)

        fig.update_layout(
            title=f"{int((1-result.alpha)*100)}% 信頼区間 — {label}",
            xaxis_title="値",
            yaxis=dict(visible=False, range=[-1, 1]),
            showlegend=True,
            legend=dict(bgcolor=_BG, bordercolor=_BORDER, borderwidth=1),
        )
    return _apply(fig, 260)


# ── Power curve ───────────────────────────────────────────────────────────────
def plot_power_curve(alpha: float, effect_size: float, n_range: np.ndarray,
                     tail: str = "two", dist: str = "t") -> go.Figure:
    from utils.tests import compute_power
    powers = [compute_power(effect_size, n, alpha, tail, dist) for n in n_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_range, y=powers, mode="lines",
        line=dict(color=_GREEN, width=2.5),
        name="検出力 (1−β)",
        fill="tozeroy", fillcolor="rgba(63,185,80,0.1)",
    ))
    # 0.8 power reference
    fig.add_hline(y=0.8, line_dash="dash", line_color=_ORANGE, line_width=1.5,
                  annotation_text="推奨 80%", annotation_font_color=_ORANGE)
    fig.add_hline(y=alpha, line_dash="dot", line_color=_RED, line_width=1.2,
                  annotation_text=f"α={alpha}", annotation_font_color=_RED)

    fig.update_layout(
        title=f"検出力曲線（効果量 d={effect_size:.2f}, α={alpha}）",
        xaxis_title="サンプルサイズ n",
        yaxis_title="検出力 (1−β)",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(bgcolor=_BG, bordercolor=_BORDER, borderwidth=1),
    )
    return _apply(fig, 380)


# ── Type I / II error illustration ───────────────────────────────────────────
def plot_error_types(mu0: float, mu1: float, sigma: float,
                     n: int, alpha: float) -> go.Figure:
    se = sigma / np.sqrt(n)
    x_min = min(mu0, mu1) - 4*se
    x_max = max(mu0, mu1) + 4*se
    x = np.linspace(x_min, x_max, 600)

    y0 = stats.norm.pdf(x, mu0, se)
    y1 = stats.norm.pdf(x, mu1, se)
    crit = mu0 + stats.norm.ppf(1 - alpha) * se  # one-sided right

    fig = go.Figure()

    # H0 distribution
    fig.add_trace(go.Scatter(x=x, y=y0, mode="lines",
                             line=dict(color=_BLUE, width=2),
                             name=f"H₀: μ={mu0}", fill="tozeroy",
                             fillcolor="rgba(88,166,255,0.08)"))
    # H1 distribution
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines",
                             line=dict(color=_GREEN, width=2),
                             name=f"H₁: μ={mu1}", fill="tozeroy",
                             fillcolor="rgba(63,185,80,0.08)"))

    # Type I error (α) — right tail of H0
    mask_a = x >= crit
    fig.add_trace(go.Scatter(
        x=np.concatenate([[x[mask_a][0]], x[mask_a], [x[mask_a][-1]]]),
        y=np.concatenate([[0], y0[mask_a], [0]]),
        fill="toself", mode="none",
        fillcolor="rgba(248,81,73,0.45)", name="第一種誤り α"))

    # Type II error (β) — left tail of H1
    mask_b = x <= crit
    fig.add_trace(go.Scatter(
        x=np.concatenate([[x[mask_b][0]], x[mask_b], [x[mask_b][-1]]]),
        y=np.concatenate([[0], y1[mask_b], [0]]),
        fill="toself", mode="none",
        fillcolor="rgba(163,113,247,0.45)", name="第二種誤り β"))

    # Critical value
    fig.add_vline(x=crit, line_dash="dash", line_color=_RED, line_width=1.5,
                  annotation_text=f"臨界値={crit:.2f}",
                  annotation_font_color=_RED)

    fig.update_layout(
        title="第一種誤り (α) と第二種誤り (β) の関係",
        xaxis_title="標本平均",
        yaxis_title="確率密度",
        legend=dict(bgcolor=_BG, bordercolor=_BORDER, borderwidth=1),
    )
    return _apply(fig, 420)


# ── Data distribution ─────────────────────────────────────────────────────────
def plot_data_distribution(data_dict: dict) -> go.Figure:
    """Overlay histograms for one or two groups."""
    fig = go.Figure()
    colors = [_BLUE, _GREEN]
    for i, (label, data) in enumerate(data_dict.items()):
        fig.add_trace(go.Histogram(
            x=data, name=label,
            marker_color=colors[i % 2],
            opacity=0.65,
            nbinsx=20,
        ))
    fig.update_layout(
        barmode="overlay",
        title="データの分布",
        xaxis_title="値",
        yaxis_title="度数",
        legend=dict(bgcolor=_BG, bordercolor=_BORDER, borderwidth=1),
    )
    return _apply(fig, 340)


# ── Contingency table heatmap ─────────────────────────────────────────────────
def plot_contingency(observed: np.ndarray, expected: np.ndarray,
                     row_labels, col_labels) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["観測度数", "期待度数"])

    def heatmap(mat, row, col, colorscale):
        fig.add_trace(
            go.Heatmap(z=mat, x=col_labels, y=row_labels,
                       text=np.round(mat, 2),
                       texttemplate="%{text}",
                       colorscale=colorscale,
                       showscale=False),
            row=row, col=col)

    heatmap(observed, 1, 1, [[0,"#0d1117"],[0.5,"#1f6feb"],[1,"#58a6ff"]])
    heatmap(expected, 1, 2, [[0,"#0d1117"],[0.5,"#238636"],[1,"#3fb950"]])

    fig.update_layout(title="分割表ヒートマップ", **_BASE, height=360)
    return fig
