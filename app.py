import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

from utils import tests as T
from utils.visualizations import (
    plot_test_distribution, plot_confidence_interval,
    plot_power_curve, plot_error_types,
    plot_data_distribution, plot_contingency,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="統計的仮説検定ツール", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}

.hero {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 60%, #161b22 100%);
    border: 1px solid #30363d; border-radius: 16px;
    padding: 26px 34px; margin-bottom: 18px; position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:0;left:0;right:0;bottom:0;
    background: radial-gradient(ellipse at 10% 50%, rgba(163,113,247,.12) 0%,transparent 55%),
                radial-gradient(ellipse at 90% 50%, rgba(88,166,255,.1) 0%,transparent 55%);
    pointer-events:none;
}
.hero-title {
    font-size:2rem; font-weight:800; margin:0 0 6px 0;
    background: linear-gradient(135deg,#a371f7 0%,#58a6ff 50%,#3fb950 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub { color:#8b949e; font-size:.92rem; }

/* Result card */
.result-card {
    border-radius:12px; padding:20px 24px; margin:12px 0;
    border:1px solid;
}
.result-card.reject {
    background:rgba(63,185,80,.08); border-color:rgba(63,185,80,.4);
}
.result-card.accept {
    background:rgba(248,81,73,.08); border-color:rgba(248,81,73,.35);
}
.result-title { font-size:1.25rem; font-weight:700; margin-bottom:10px; }
.result-title.reject { color:#3fb950; }
.result-title.accept { color:#f85149; }

/* Stat grid */
.sg { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
      gap:12px; margin:14px 0; }
.sc { background:#161b22; border:1px solid #30363d; border-radius:10px;
      padding:14px; text-align:center; }
.sc .lbl { font-size:.66rem; color:#8b949e; text-transform:uppercase;
           letter-spacing:1.4px; margin-bottom:6px; }
.sc .val { font-size:1.5rem; font-weight:700; }
.sc .val.b { color:#58a6ff; }
.sc .val.g { color:#3fb950; }
.sc .val.r { color:#f85149; }
.sc .val.o { color:#f0883e; }
.sc .val.p { color:#a371f7; }

/* Section header */
.sh { display:flex; align-items:center; gap:8px; margin:22px 0 10px 0; }
.sh .ic { font-size:1.1rem; }
.sh .tx { font-size:1rem; font-weight:600; color:#e6edf3; }
.sh::after { content:''; flex:1; height:1px; background:#30363d; }

/* Hypothesis box */
.hypo { background:#161b22; border:1px solid #30363d; border-radius:10px;
        padding:14px 18px; margin:8px 0; font-family:monospace; }
.hypo .h0 { color:#8b949e; margin-bottom:4px; }
.hypo .h1 { color:#58a6ff; }

/* Info box */
.ib { background:rgba(31,111,235,.1); border:1px solid rgba(31,111,235,.3);
      border-radius:10px; padding:12px 16px; color:#79c0ff; font-size:.88rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#161b22 0%,#0d1117 100%);
    border-right:1px solid #30363d;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:#161b22; border:1px solid #30363d;
    border-radius:10px; padding:4px; gap:2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px !important; color:#8b949e !important;
    padding:8px 16px !important; background:transparent !important; border:none !important;
}
.stTabs [aria-selected="true"] { background:#1f6feb !important; color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display:none; }

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#6e40c9,#a371f7) !important;
    color:#fff !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#8957e5,#bf8fff) !important;
    transform:translateY(-1px);
    box-shadow:0 4px 16px rgba(110,64,201,.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def sc(label, value, cls="b"):
    return f'<div class="sc"><div class="lbl">{label}</div><div class="val {cls}">{value}</div></div>'

def sh(icon, title):
    return f'<div class="sh"><span class="ic">{icon}</span><span class="tx">{title}</span></div>'

def parse_array(text: str) -> np.ndarray:
    """Parse comma/newline separated numbers into ndarray."""
    import re
    nums = re.split(r"[,\s\n]+", text.strip())
    return np.array([float(x) for x in nums if x])

TESTS = {
    "1標本 t 検定": "one_t",
    "2標本 t 検定": "two_t",
    "対応のある t 検定": "paired_t",
    "z 検定（母分散既知）": "z_test",
    "カイ二乗検定（独立性）": "chi2",
}

SAMPLE_DATA = {
    "one_t":    {"説明": "ある授業の平均点が70点かを検定",
                 "data1": "72,68,75,80,65,70,78,73,69,74,71,76,68,72,77",
                 "mu0": 70.0},
    "two_t":    {"説明": "2グループの平均を比較",
                 "data1": "72,68,75,80,65,70,78,73",
                 "data2": "65,62,70,68,71,64,67,69"},
    "paired_t": {"説明": "治療前後の測定値を比較",
                 "data1": "120,130,125,118,135,128,122,131",
                 "data2": "115,122,118,112,127,120,116,124"},
    "z_test":   {"説明": "母標準偏差既知の場合の平均検定",
                 "data1": "72,68,75,80,65,70,78,73,69,74",
                 "sigma": 8.0, "mu0": 70.0},
    "chi2":     {"説明": "性別×支持政党の独立性を検定",
                 "table": "30,20\n25,25"},
}

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🔬 統計的仮説検定ツール</div>
  <div class="hero-sub">DS02 で学んだ検定手法をインタラクティブに体験 — 分布・棄却域・検出力をリアルタイム可視化</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 検定設定")
    test_label = st.selectbox("検定手法", list(TESTS.keys()))
    test_key   = TESTS[test_label]

    st.divider()
    alpha = st.select_slider("有意水準 α", options=[0.01, 0.05, 0.10], value=0.05)

    if test_key in ("one_t", "two_t", "paired_t", "z_test"):
        tail = st.radio("対立仮説の向き", ["two", "left", "right"],
                        format_func=lambda x: {"two":"両側","left":"左側","right":"右側"}[x])
    else:
        tail = "right"

    if test_key == "two_t":
        equal_var = st.toggle("等分散を仮定", value=True)

    st.divider()
    use_sample = st.toggle("サンプルデータを使う", value=True)

    if use_sample:
        st.markdown(f'<div class="ib" style="margin-top:4px">📌 {SAMPLE_DATA[test_key]["説明"]}</div>',
                    unsafe_allow_html=True)

# ── Data Input Section ────────────────────────────────────────────────────────
st.markdown(sh("📥", "データ入力"), unsafe_allow_html=True)

result = None
expected = None
data_dict = {}
contingency = None

# --- Non-chi2 tests ---
if test_key != "chi2":
    col1, col2 = st.columns(2 if test_key in ("two_t", "paired_t") else [1, 1])

    with col1:
        default1 = SAMPLE_DATA[test_key]["data1"] if use_sample else ""
        raw1 = st.text_area("グループ 1（カンマ区切り）", value=default1, height=100)

    need_data2 = test_key in ("two_t", "paired_t")
    if need_data2:
        with col2:
            default2 = SAMPLE_DATA[test_key].get("data2", "") if use_sample else ""
            raw2 = st.text_area("グループ 2（カンマ区切り）", value=default2, height=100)

    c1, c2, c3 = st.columns(3)
    if test_key in ("one_t", "z_test"):
        with c1:
            mu0 = st.number_input("帰無仮説の平均 μ₀", value=float(SAMPLE_DATA[test_key]["mu0"]))
    if test_key == "z_test":
        with c2:
            sigma = st.number_input("母標準偏差 σ（既知）",
                                    value=float(SAMPLE_DATA[test_key].get("sigma", 10.0)),
                                    min_value=0.01)

# --- Chi2 ---
else:
    st.markdown('<div class="ib">行×列の分割表を入力してください（カンマ区切り、1行=1行）</div>',
                unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        default_tbl = SAMPLE_DATA["chi2"]["table"] if use_sample else "10,20\n30,40"
        raw_tbl = st.text_area("分割表", value=default_tbl, height=100)
        n_rows = st.number_input("行数", min_value=2, max_value=10, value=2)
        n_cols = st.number_input("列数", min_value=2, max_value=10, value=2)
    with col_r:
        row_lbls = st.text_input("行ラベル（カンマ区切り）", value="グループA,グループB")
        col_lbls = st.text_input("列ラベル（カンマ区切り）", value="カテゴリ1,カテゴリ2")

# ── Run Button ────────────────────────────────────────────────────────────────
st.markdown("")
if st.button("▶ 検定を実行", type="primary", use_container_width=False):
    try:
        if test_key == "one_t":
            data1 = parse_array(raw1)
            data_dict = {"データ": data1}
            result = T.one_sample_t(data1, mu0, alpha, tail)

        elif test_key == "two_t":
            data1, data2 = parse_array(raw1), parse_array(raw2)
            data_dict = {"グループ1": data1, "グループ2": data2}
            result = T.two_sample_t(data1, data2, alpha, tail, equal_var)

        elif test_key == "paired_t":
            data1, data2 = parse_array(raw1), parse_array(raw2)
            if len(data1) != len(data2):
                st.error("対応のある検定: データ数が一致している必要があります")
                st.stop()
            data_dict = {"グループ1": data1, "グループ2": data2}
            result = T.paired_t(data1, data2, alpha, tail)

        elif test_key == "z_test":
            data1 = parse_array(raw1)
            data_dict = {"データ": data1}
            result = T.z_test(data1, mu0, sigma, alpha, tail)

        elif test_key == "chi2":
            rows = [list(map(float, row.split(","))) for row in raw_tbl.strip().split("\n")]
            contingency = np.array(rows)
            result, expected = T.chi2_independence(contingency, alpha)

        st.session_state["result"] = result
        st.session_state["expected"] = expected
        st.session_state["data_dict"] = data_dict
        st.session_state["contingency"] = contingency
        st.session_state["test_key"] = test_key

    except Exception as e:
        st.error(f"エラー: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.get("result"):
    result      = st.session_state["result"]
    expected    = st.session_state["expected"]
    data_dict   = st.session_state["data_dict"]
    contingency = st.session_state["contingency"]
    test_key_s  = st.session_state["test_key"]

    st.divider()

    # ── Decision card
    verdict_cls   = "reject" if result.reject_h0 else "accept"
    verdict_icon  = "✅" if result.reject_h0 else "❌"
    verdict_text  = "帰無仮説を棄却する" if result.reject_h0 else "帰無仮説を棄却できない"
    verdict_color = "#3fb950" if result.reject_h0 else "#f85149"

    st.markdown(f"""
    <div class="result-card {verdict_cls}">
      <div class="result-title {verdict_cls}">{verdict_icon} {verdict_text}</div>
      <div style="color:#8b949e;font-size:.88rem">
        p値 = <b style="color:{verdict_color}">{result.p_value:.4f}</b>
        {'<' if result.reject_h0 else '≥'}
        α = {result.alpha}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat cards
    cards = [sc("検定統計量", f"{result.statistic:.4f}", "o")]
    if result.df is not None:
        cards.append(sc("自由度 df", f"{result.df:.1f}", "p"))
    cards.append(sc("p 値", f"{result.p_value:.4f}", "g" if result.reject_h0 else "r"))
    cards.append(sc("有意水準 α", result.alpha, "b"))
    if result.n:
        cards.append(sc("n₁", result.n, "b"))
    if result.n2:
        cards.append(sc("n₂", result.n2, "b"))
    if result.effect_size is not None:
        cards.append(sc("効果量 d", f"{result.effect_size:.3f}", "p"))

    st.markdown(f'<div class="sg">{"".join(cards)}</div>', unsafe_allow_html=True)

    # ── Hypothesis box
    st.markdown(f"""
    <div class="hypo">
      <div class="h0">H₀: {result.h0}</div>
      <div class="h1">H₁: {result.h1}</div>
    </div>
    """, unsafe_allow_html=True)

    if result.ci:
        st.markdown(f"""
        <div class="ib" style="margin-top:8px">
          📏 {int((1-result.alpha)*100)}% 信頼区間:
          <b>[ {result.ci[0]:.4f}, {result.ci[1]:.4f} ]</b>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs
    tab_labels = ["📊 分布・棄却域", "📏 区間推定", "⚡ 検出力分析"]
    if test_key_s == "chi2":
        tab_labels = ["📊 分布・棄却域", "🔲 分割表"]
    tabs = st.tabs(tab_labels)

    # Tab 1: distribution
    with tabs[0]:
        col_d, col_h = st.columns([3, 2])
        with col_d:
            st.markdown(sh("📈", "データの分布"), unsafe_allow_html=True)
            if data_dict:
                st.plotly_chart(plot_data_distribution(data_dict), use_container_width=True)
            elif contingency is not None:
                st.plotly_chart(plot_contingency(
                    contingency, expected,
                    [r.strip() for r in row_lbls.split(",")],
                    [c.strip() for c in col_lbls.split(",")],
                ), use_container_width=True)
        with col_h:
            st.markdown(sh("📐", "記述統計"), unsafe_allow_html=True)
            if data_dict:
                rows = []
                for lbl, d in data_dict.items():
                    rows.append({
                        "グループ": lbl, "n": len(d),
                        "平均": f"{np.mean(d):.3f}",
                        "標準偏差": f"{np.std(d, ddof=1):.3f}",
                        "中央値": f"{np.median(d):.3f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown(sh("📉", "検定分布と棄却域"), unsafe_allow_html=True)
        st.plotly_chart(plot_test_distribution(result), use_container_width=True)

    # Tab 2: CI or contingency
    with tabs[1]:
        if test_key_s == "chi2":
            if expected is not None:
                r_lbls = [r.strip() for r in row_lbls.split(",")]
                c_lbls = [c.strip() for c in col_lbls.split(",")]
                st.markdown(sh("🔲", "観測度数 vs 期待度数"), unsafe_allow_html=True)
                st.plotly_chart(plot_contingency(contingency, expected, r_lbls, c_lbls),
                                use_container_width=True)
                st.markdown(sh("📋", "残差分析"), unsafe_allow_html=True)
                residuals = (contingency - expected) / np.sqrt(expected)
                df_res = pd.DataFrame(np.round(residuals, 3),
                                      index=r_lbls, columns=c_lbls)
                st.dataframe(df_res, use_container_width=True)
        else:
            if result.ci:
                st.markdown(sh("📏", "信頼区間の可視化"), unsafe_allow_html=True)
                st.plotly_chart(plot_confidence_interval(result), use_container_width=True)

                # CI interpretation
                in_ci = result.ci[0] <= 0 <= result.ci[1] if test_key_s in ("two_t","paired_t") else True
                try:
                    h0_v = float(result.h0.split("=")[1].strip())
                    in_ci = result.ci[0] <= h0_v <= result.ci[1]
                except Exception:
                    pass

                st.markdown(f"""
                <div class="ib">
                {'⚠️ H₀の値が信頼区間 <b>外</b> → 帰無仮説を棄却' if not in_ci
                 else '✅ H₀の値が信頼区間 <b>内</b> → 帰無仮説を棄却できない'}
                </div>
                """, unsafe_allow_html=True)

    # Tab 3: Power
    if test_key_s != "chi2":
        with tabs[2]:
            st.markdown(sh("⚡", "検出力分析"), unsafe_allow_html=True)
            col_p, col_e = st.columns([3, 2])

            with col_e:
                d_input = st.slider("効果量 d", 0.1, 2.0,
                                    float(abs(result.effect_size)) if result.effect_size else 0.5,
                                    0.05)
                n_max   = st.slider("サンプルサイズ最大値", 20, 500, 200, 10)
                n_range = np.arange(5, n_max + 1)

                dist_str = "t" if result.dist_type == "t" else "normal"
                current_power = T.compute_power(d_input, result.n or 30, result.alpha,
                                                result.tail, dist_str)
                n80 = next((n for n in n_range
                            if T.compute_power(d_input, n, result.alpha, result.tail, dist_str) >= 0.8),
                           None)

                st.markdown(f"""
                <div class="sg" style="margin-top:16px">
                  {sc("現在の検出力", f"{current_power:.1%}", "g" if current_power>=0.8 else "r")}
                  {sc("80%達成に必要な n", str(n80) if n80 else "範囲外", "o")}
                </div>
                """, unsafe_allow_html=True)

            with col_p:
                st.plotly_chart(plot_power_curve(result.alpha, d_input, n_range,
                                                 result.tail, dist_str),
                                use_container_width=True)

            # Error type illustration (only for parametric tests with means)
            if result.mean1 is not None and result.mean2 is not None and test_key_s in ("two_t", "paired_t"):
                st.markdown(sh("🎯", "第一種誤り・第二種誤りの可視化"), unsafe_allow_html=True)
                sigma_est = result.std1 or 1.0
                n_err = result.n or 30
                st.plotly_chart(
                    plot_error_types(result.mean2, result.mean1, sigma_est, n_err, result.alpha),
                    use_container_width=True)
            elif result.mean1 is not None and test_key_s == "one_t":
                st.markdown(sh("🎯", "第一種誤り・第二種誤りの可視化"), unsafe_allow_html=True)
                try:
                    mu0_v = float(result.h0.split("=")[1].strip())
                except Exception:
                    mu0_v = result.mean1 - 1.0
                st.plotly_chart(
                    plot_error_types(mu0_v, result.mean1, result.std1 or 1.0,
                                    result.n or 30, result.alpha),
                    use_container_width=True)

else:
    st.markdown('<div class="ib" style="margin-top:16px">👆 パラメータを設定して「検定を実行」ボタンを押してください</div>',
                unsafe_allow_html=True)
