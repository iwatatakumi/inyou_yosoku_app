# -*- coding: utf-8 -*-
"""
豊田通商(8015) 陰陽予測ダッシュボード
Streamlit アプリケーション
"""
import os
import sys
import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# カレントディレクトリをスクリプトのあるフォルダに設定
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="陰陽予測ダッシュボード | 豊田通商(8015)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# カスタム CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── 全体 ───────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── ヘッダー ───────────────────────── */
.dashboard-header {
    background: linear-gradient(135deg, #1a1f35 0%, #0d2137 50%, #0a1628 100%);
    border: 1px solid #1f6feb;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    box-shadow: 0 4px 24px rgba(31, 111, 235, 0.15);
}
.dashboard-title {
    font-size: 2.0rem;
    font-weight: 800;
    background: linear-gradient(90deg, #58a6ff, #79c0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 4px 0;
}
.dashboard-sub {
    font-size: 0.95rem;
    color: #8b949e;
    margin: 0;
}

/* ── 予測カード（アンサンブル） ─────── */
.card-yosen {
    background: linear-gradient(135deg, #1a0a0a 0%, #3d0000 100%);
    border: 1px solid #f85149;
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(248, 81, 73, 0.25);
}
.card-insen {
    background: linear-gradient(135deg, #0a0d1a 0%, #00153d 100%);
    border: 1px solid #388bfd;
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(56, 139, 253, 0.25);
}
.card-label {
    font-size: 1.0rem;
    color: #8b949e;
    margin-bottom: 8px;
    letter-spacing: 0.05em;
}
.card-main-value {
    font-size: 3.5rem;
    font-weight: 900;
    line-height: 1;
    margin: 8px 0;
}
.card-main-value.yosen { color: #ff7b72; }
.card-main-value.insen { color: #58a6ff; }
.card-sub-value {
    font-size: 1.1rem;
    color: #8b949e;
    margin-top: 6px;
}
.card-vote {
    font-size: 0.85rem;
    margin-top: 12px;
    color: #c9d1d9;
    letter-spacing: 0.03em;
}

/* ── モデルカード ─────────────────── */
.model-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 18px;
    text-align: center;
    height: 100%;
}
.model-card.yosen { border-color: #f85149; }
.model-card.insen { border-color: #388bfd; }
.model-name {
    font-size: 0.8rem;
    color: #8b949e;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.model-pred-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 4px 0;
}
.model-pred-value.yosen { color: #ff7b72; }
.model-pred-value.insen { color: #58a6ff; }
.model-label {
    font-size: 0.9rem;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 999px;
    display: inline-block;
    margin-top: 6px;
}
.model-label.yosen { background: rgba(248,81,73,0.15); color: #ff7b72; }
.model-label.insen { background: rgba(56,139,253,0.15); color: #58a6ff; }
.model-rmse {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 10px;
}

/* ── メトリクスセクション ─────────── */
.section-title {
    font-size: 1.0rem;
    font-weight: 700;
    color: #c9d1d9;
    letter-spacing: 0.05em;
    padding: 6px 0;
    border-bottom: 1px solid #30363d;
    margin-bottom: 16px;
}

/* ── 投票ドット ─────────────────── */
.vote-dot-yosen {
    display: inline-block;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #ff7b72;
    margin: 0 3px;
    box-shadow: 0 0 6px rgba(255,123,114,0.6);
}
.vote-dot-insen {
    display: inline-block;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #388bfd;
    margin: 0 3px;
    box-shadow: 0 0 6px rgba(56,139,253,0.6);
}

/* ── Streamlit メトリクス上書き ─── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.6rem !important; }

/* ── ボタン ─────────────────────── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── セパレータ ─────────────────── */
hr { border-color: #30363d !important; }

/* ── expander ─────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    background: #161b22 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ヘルパー関数
# ─────────────────────────────────────────────
def _label_class(label: str) -> str:
    return "yosen" if label == "陽線" else "insen"


def _vote_dots(vote_count: int, total: int = 3) -> str:
    dots = ""
    for i in range(total):
        dots += '<span class="vote-dot-yosen"></span>' if i < vote_count else '<span class="vote-dot-insen"></span>'
    return dots


def _fmt_value(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.4f}%"


def render_result(result: dict):
    """予測結果を画面に描画する"""
    pred = result["prediction"]
    metrics = result["metrics"]
    fi = result["feature_importance"]
    meta = result["meta"]

    ens = pred["ensemble"]
    ens_class = _label_class(ens["label"])
    vote = ens["vote_count"]

    # ── ヘッダー ─────────────────────────────────────────
    pred_date = meta["prediction_date"]
    elapsed = meta["elapsed_seconds"]
    st.markdown(f"""
    <div class="dashboard-header">
        <p class="dashboard-title">📈 豊田通商(8015) 陰陽予測ダッシュボード</p>
        <p class="dashboard-sub">予測対象日: {pred_date}　｜　学習データ: {meta['yf_start']} ～　｜　サンプル数: {meta['train_data_count']:,}件　｜　処理時間: {elapsed:.1f}秒</p>
    </div>
    """, unsafe_allow_html=True)

    # ── アンサンブル予測（メイン） ─────────────────────────
    st.markdown('<p class="section-title">🔮 アンサンブル予測（3モデル総合）</p>', unsafe_allow_html=True)
    col_main, col_side = st.columns([1, 2])

    with col_main:
        vote_html = _vote_dots(vote)
        st.markdown(f"""
        <div class="card-{ens_class}">
            <p class="card-label">本日の予測</p>
            <p class="card-main-value {ens_class}">{ens['label']}</p>
            <p class="card-sub-value">{_fmt_value(ens['value'])}</p>
            <p class="card-vote">
                モデル投票 &nbsp; {vote_html} &nbsp; 陽線 {vote} / {3 - vote} 陰線
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_side:
        # 予測値ゲージ
        gauge_max = 1.5
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ens["value"],
            number={"suffix": "%", "font": {"size": 28, "color": "#ff7b72" if ens["value"] > 0 else "#388bfd"}},
            delta={"reference": 0, "relative": False, "valueformat": ".4f"},
            gauge={
                "axis": {"range": [-gauge_max, gauge_max], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e", "size": 10}},
                "bar": {"color": "#ff7b72" if ens["value"] > 0 else "#388bfd", "thickness": 0.3},
                "bgcolor": "#161b22",
                "borderwidth": 0,
                "steps": [
                    {"range": [-gauge_max, 0], "color": "rgba(56,139,253,0.1)"},
                    {"range": [0, gauge_max], "color": "rgba(248,81,73,0.1)"},
                ],
                "threshold": {
                    "line": {"color": "#e6edf3", "width": 2},
                    "thickness": 0.8,
                    "value": 0,
                },
            },
            title={"text": "予測値スケール", "font": {"color": "#8b949e", "size": 13}},
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font={"color": "#e6edf3"},
            height=220,
            margin=dict(t=40, b=10, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 各モデル予測 ──────────────────────────────────────
    st.markdown('<p class="section-title">🤖 モデル別予測結果</p>', unsafe_allow_html=True)
    model_defs = [
        ("LightGBM", pred["lgbm"], metrics["lgbm_rmse"]),
        ("XGBoost",  pred["xgb"],  metrics["xgb_rmse"]),
        ("RandomForest", pred["rft"], metrics["rft_rmse"]),
    ]
    cols = st.columns(3)
    for col, (name, p, rmse) in zip(cols, model_defs):
        cls = _label_class(p["label"])
        with col:
            st.markdown(f"""
            <div class="model-card {cls}">
                <p class="model-name">{name}</p>
                <p class="model-pred-value {cls}">{p['label']}</p>
                <p class="model-pred-value {cls}" style="font-size:1.3rem;">{_fmt_value(p['value'])}</p>
                <span class="model-label {cls}">{p['label']}</span>
                <p class="model-rmse">RMSE: {rmse:.4f}　有効モデル数: {p['valid_model_count']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 精度メトリクス ────────────────────────────────────
    st.markdown('<p class="section-title">📊 モデル精度（RMSE）</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    best_rmse = min(metrics["lgbm_rmse"], metrics["xgb_rmse"], metrics["rft_rmse"])
    with c1:
        st.metric("LightGBM  RMSE", f"{metrics['lgbm_rmse']:.4f}",
                  delta="最良" if metrics["lgbm_rmse"] == best_rmse else None)
    with c2:
        st.metric("XGBoost  RMSE", f"{metrics['xgb_rmse']:.4f}",
                  delta="最良" if metrics["xgb_rmse"] == best_rmse else None)
    with c3:
        st.metric("RandomForest  RMSE", f"{metrics['rft_rmse']:.4f}",
                  delta="最良" if metrics["rft_rmse"] == best_rmse else None)
    with c4:
        avg_rmse = (metrics["lgbm_rmse"] + metrics["xgb_rmse"] + metrics["rft_rmse"]) / 3
        st.metric("平均 RMSE", f"{avg_rmse:.4f}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 特徴量重要度 ──────────────────────────────────────
    st.markdown('<p class="section-title">🔍 特徴量重要度（LightGBM Gain 上位15件）</p>', unsafe_allow_html=True)
    top_n = 15
    feats = fi["features"][:top_n]
    gains = fi["gains"][:top_n]

    fig2 = go.Figure(go.Bar(
        x=gains[::-1],
        y=feats[::-1],
        orientation="h",
        marker=dict(
            color=gains[::-1],
            colorscale=[[0, "#1f6feb"], [0.5, "#58a6ff"], [1, "#ff7b72"]],
            showscale=False,
        ),
        text=[f"{v:.1f}" for v in gains[::-1]],
        textposition="outside",
        textfont=dict(color="#8b949e", size=11),
    ))
    fig2.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font={"color": "#e6edf3", "size": 12},
        xaxis=dict(
            title="Gain（重要度）",
            gridcolor="#30363d",
            color="#8b949e",
            showline=False,
        ),
        yaxis=dict(
            gridcolor="#30363d",
            color="#c9d1d9",
            showline=False,
            tickfont=dict(size=11),
        ),
        margin=dict(l=160, r=60, t=20, b=40),
        height=420,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 詳細情報（折りたたみ） ─────────────────────────────
    with st.expander("🔧 詳細情報・生データ"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**モデル別予測値**")
            import pandas as pd
            df_pred = pd.DataFrame({
                "モデル": ["LightGBM", "XGBoost", "RandomForest", "アンサンブル"],
                "予測値 (%)": [
                    pred["lgbm"]["value"],
                    pred["xgb"]["value"],
                    pred["rft"]["value"],
                    pred["ensemble"]["value"],
                ],
                "判定": [
                    pred["lgbm"]["label"],
                    pred["xgb"]["label"],
                    pred["rft"]["label"],
                    pred["ensemble"]["label"],
                ],
                "有効モデル数": [
                    pred["lgbm"]["valid_model_count"],
                    pred["xgb"]["valid_model_count"],
                    pred["rft"]["valid_model_count"],
                    "-",
                ],
            })
            st.dataframe(df_pred, use_container_width=True, hide_index=True)
        with c2:
            st.write("**メタ情報**")
            st.json({
                "予測対象日": meta["prediction_date"],
                "学習開始日": meta["yf_start"],
                "学習サンプル数": meta["train_data_count"],
                "処理時間(秒)": round(meta["elapsed_seconds"], 1),
            })


# ─────────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    st.markdown("---")

    yf_start = st.date_input(
        "データ取得開始日",
        value=datetime.date(2020, 1, 1),
        min_value=datetime.date(2015, 1, 1),
        max_value=datetime.date.today() - datetime.timedelta(days=365),
        help="学習データの取得開始日です。最低1年以上前の日付を指定してください（推奨: 2020-01-01）",
    )

    # データ期間チェック
    days_range = (datetime.date.today() - yf_start).days
    if days_range < 365:
        st.warning("⚠️ 開始日が直近すぎます。\n2020-01-01 など1年以上前を指定してください。")

    st.markdown("---")
    run_button = st.button("🚀  予測を実行", use_container_width=True, disabled=(days_range < 365))
    st.markdown("---")

    st.markdown("""
    <div style="color:#8b949e; font-size:0.78rem; line-height:1.6;">
    <b>モデル構成</b><br>
    ・LightGBM（勾配ブースティング）<br>
    ・XGBoost（勾配ブースティング）<br>
    ・RandomForest（アンサンブル）<br><br>
    <b>検証方法</b>: KFold（5分割）<br>
    <b>目的変数</b>: (終値-始値)/終値×100<br>
    &nbsp;&nbsp;→ + = 陽線 / − = 陰線<br><br>
    <b>データソース</b>: yfinance<br>
    &nbsp;&nbsp;・豊田通商 (8015.T)<br>
    &nbsp;&nbsp;・S&amp;P500 / NYダウ / NASDAQ<br>
    &nbsp;&nbsp;・VIX / USD/JPY<br>
    &nbsp;&nbsp;・香港ハンセン / 上海総合
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# メインエリア（初期表示 / 実行中 / 結果）
# ─────────────────────────────────────────────
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if run_button:
    # ── 実行中 UI ───────────────────────────
    st.markdown(f"""
    <div class="dashboard-header">
        <p class="dashboard-title">📈 豊田通商(8015) 陰陽予測ダッシュボード</p>
        <p class="dashboard-sub">機械学習モデルを学習中です。しばらくお待ちください…</p>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def on_progress(step, total, message):
        pct = int(step / total * 100)
        progress_bar.progress(pct)
        status_text.markdown(
            f"<span style='color:#8b949e; font-size:0.9rem;'>⏳ {message}（{step}/{total}）</span>",
            unsafe_allow_html=True,
        )

    from predict import run_prediction
    result = run_prediction(
        yf_start=yf_start.strftime("%Y-%m-%d"),
        progress_callback=on_progress,
    )

    progress_bar.progress(100)
    status_text.markdown(
        "<span style='color:#3fb950; font-size:0.9rem;'>✅ 予測完了！</span>",
        unsafe_allow_html=True,
    )

    if result["meta"]["error"]:
        st.error(f"エラーが発生しました：\n\n```\n{result['meta']['error']}\n```")
    else:
        st.session_state["last_result"] = result
        st.rerun()

elif st.session_state["last_result"] is not None:
    render_result(st.session_state["last_result"])

else:
    # ── 初期画面 ─────────────────────────────
    st.markdown(f"""
    <div class="dashboard-header">
        <p class="dashboard-title">📈 豊田通商(8015) 陰陽予測ダッシュボード</p>
        <p class="dashboard-sub">左のサイドバーから「予測を実行」ボタンを押すと予測が開始されます</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: #161b22;
        border: 1px dashed #30363d;
        border-radius: 16px;
        padding: 60px;
        text-align: center;
        margin-top: 40px;
    ">
        <p style="font-size: 3rem; margin: 0;">📊</p>
        <p style="font-size: 1.4rem; color: #c9d1d9; margin: 16px 0 8px 0; font-weight: 700;">
            陰陽予測ダッシュボード
        </p>
        <p style="color: #8b949e; font-size: 0.95rem; line-height: 1.7;">
            3つの機械学習モデル（LightGBM・XGBoost・RandomForest）を使用して<br>
            豊田通商(8015)の翌営業日の陰陽を予測します。<br><br>
            ← 左のサイドバーで開始日を設定し、「予測を実行」を押してください。<br>
            <span style="color: #8b949e; font-size: 0.85rem;">
                ※ 初回実行は学習に数分かかります
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
