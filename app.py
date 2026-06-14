import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import numpy as np
import time

st.set_page_config(
    page_title="⛳ ゴルフ場 精密天気予報",
    page_icon="⛳",
    layout="wide"
)

# ---- スタイル ----
st.markdown("""
<style>
body { font-family: 'Hiragino Sans', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #1a3a2a 0%, #2d6a4f 100%);
    border-radius: 12px; padding: 16px; color: white; text-align: center;
    margin: 4px;
}
.metric-val { font-size: 28px; font-weight: bold; }
.metric-label { font-size: 12px; opacity: 0.8; }
.accuracy-badge {
    background: #f0a500; color: #000; border-radius: 20px;
    padding: 4px 14px; font-size: 13px; font-weight: bold;
    display: inline-block; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---- 気象モデル定義 ----
WEATHER_MODELS = {
    "ECMWF IFS (欧州中期予報センター)": "ecmwf_ifs025",
    "GFS (米NOAA)": "gfs_seamless",
    "ICON (独DWD)": "icon_seamless",
    "GEM (カナダ)": "gem_seamless",
    "JMA (気象庁)": "jma_seamless",
    "Météo-France (仏)": "meteofrance_seamless",
    "ACCESS-G (豪BOM)": "bom_access_global",
    "ARPAE (伊)": "arpae_cosmo_2i",
}

WEATHER_CODE = {
    0: "☀️ 快晴", 1: "🌤️ 晴れ", 2: "⛅ 曇りがち", 3: "☁️ 曇り",
    45: "🌫️ 霧", 48: "🌫️ 霧氷", 51: "🌦️ 霧雨(弱)", 53: "🌦️ 霧雨",
    55: "🌧️ 霧雨(強)", 61: "🌧️ 小雨", 63: "🌧️ 雨", 65: "🌧️ 大雨",
    71: "🌨️ 小雪", 73: "🌨️ 雪", 75: "❄️ 大雪", 77: "🌨️ 霙",
    80: "🌦️ にわか雨(弱)", 81: "🌦️ にわか雨", 82: "⛈️ にわか雨(強)",
    85: "🌨️ にわか雪", 86: "❄️ にわか大雪",
    95: "⛈️ 雷雨", 96: "⛈️ 雷雨+雹", 99: "⛈️ 激しい雷雨",
}

def get_weather_code_label(code):
    return WEATHER_CODE.get(int(code) if not np.isnan(code) else 0, f"コード{int(code)}")

# ---- ゴルフ場 → 座標変換 ----
@st.cache_data(ttl=3600)
def geocode_golf_course(name: str):
    geolocator = Nominatim(user_agent="golf_weather_app_v1")
    # まず「ゴルフ場」として検索
    for query in [f"{name} ゴルフ場", f"{name} golf course", name]:
        try:
            loc = geolocator.geocode(query, timeout=10, language="ja")
            if loc:
                return loc.latitude, loc.longitude, loc.address
        except Exception:
            time.sleep(1)
    return None, None, None

# ---- 単一モデルの天気データ取得 ----
@st.cache_data(ttl=1800)
def fetch_model_forecast(lat, lon, model_id):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "weathercode",
            "windspeed_10m",
            "winddirection_10m",
            "apparent_temperature",
            "cloudcover",
            "uv_index",
            "visibility",
        ],
        "models": model_id,
        "forecast_days": 7,
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ---- アンサンブル天気データ取得（50モデル） ----
@st.cache_data(ttl=1800)
def fetch_ensemble_forecast(lat, lon):
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "precipitation_probability",
            "windspeed_10m",
            "weathercode",
        ],
        "models": "icon_seamless",  # ICON: 40メンバー
        "forecast_days": 7,
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ---- 複数モデルをアンサンブル集計 ----
def build_ensemble_df(lat, lon, models_to_use):
    all_dfs = []
    model_status = {}

    progress = st.progress(0, text="気象モデルを取得中...")
    for i, (name, model_id) in enumerate(models_to_use.items()):
        progress.progress((i + 1) / len(models_to_use), text=f"取得中: {name}")
        data = fetch_model_forecast(lat, lon, model_id)
        if data and "hourly" in data:
            df = pd.DataFrame(data["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            df["model"] = name
            all_dfs.append(df)
            model_status[name] = "✅"
        else:
            model_status[name] = "❌ (取得失敗)"
        time.sleep(0.1)

    progress.empty()

    if not all_dfs:
        return None, model_status

    # 全モデルを時刻でアライン
    numeric_cols = [
        "temperature_2m", "precipitation_probability", "precipitation",
        "windspeed_10m", "winddirection_10m", "apparent_temperature",
        "cloudcover", "uv_index", "visibility",
    ]

    # 各モデルのDataFrameを結合して時刻ごとに平均
    combined = pd.concat(all_dfs, ignore_index=True)
    agg = {col: "mean" for col in numeric_cols if col in combined.columns}
    # weathercodeはmodeで集計
    if "weathercode" in combined.columns:
        agg["weathercode"] = lambda x: x.mode()[0] if len(x) > 0 else 0

    ensemble_df = combined.groupby("time").agg(agg).reset_index()

    # 標準偏差（不確実性）を計算
    std_df = combined.groupby("time")[["temperature_2m", "precipitation"]].std().reset_index()
    std_df.columns = ["time", "temp_std", "precip_std"]
    ensemble_df = ensemble_df.merge(std_df, on="time", how="left")

    return ensemble_df, model_status

# ---- メイン UI ----
st.title("⛳ ゴルフ場 精密天気予報")
st.caption("世界8カ国の気象モデルをアンサンブル集計 — 統計的に最も精度の高い予報を提供")

col1, col2 = st.columns([3, 1])
with col1:
    course_name = st.text_input(
        "ゴルフ場名を入力",
        placeholder="例: 小野東洋ゴルフクラブ、Pebble Beach Golf Links",
        label_visibility="collapsed"
    )
with col2:
    search_btn = st.button("🔍 予報を取得", type="primary", use_container_width=True)

# モデル選択
with st.expander("⚙️ 使用する気象モデルを選択（デフォルト: 全8モデル）"):
    selected_models = {}
    cols = st.columns(4)
    for i, (name, model_id) in enumerate(WEATHER_MODELS.items()):
        with cols[i % 4]:
            if st.checkbox(name, value=True, key=f"model_{i}"):
                selected_models[name] = model_id

if not selected_models:
    st.warning("少なくとも1つのモデルを選択してください。")
    st.stop()

if search_btn and course_name:
    # --- 座標取得 ---
    with st.spinner(f"「{course_name}」の場所を特定中..."):
        lat, lon, address = geocode_golf_course(course_name)

    if lat is None:
        st.error("ゴルフ場が見つかりませんでした。正式名称や英語名で試してください。")
        st.stop()

    st.success(f"📍 **{course_name}**  |  {address}  |  緯度 {lat:.4f} / 経度 {lon:.4f}")

    # --- アンサンブル取得 ---
    ensemble_df, model_status = build_ensemble_df(lat, lon, selected_models)

    if ensemble_df is None:
        st.error("天気データの取得に失敗しました。しばらく待ってから再試行してください。")
        st.stop()

    # モデル取得状況
    with st.expander("📡 モデル取得状況"):
        for name, status in model_status.items():
            st.write(f"{status} {name}")

    # --- 今日・明日・明後日タブ ---
    now = pd.Timestamp.now(tz=ensemble_df["time"].dt.tz).tz_convert(ensemble_df["time"].dt.tz) if ensemble_df["time"].dt.tz else pd.Timestamp.now()

    tabs = st.tabs(["📅 今日", "📅 明日", "📅 明後日", "📆 3〜7日先", "📊 全期間グラフ"])

    def render_day_detail(df_day, title):
        if df_day.empty:
            st.info("データなし")
            return

        # サマリー指標
        st.markdown(f"### {title}")
        m_cols = st.columns(5)
        metrics = [
            ("🌡️ 最高気温", f"{df_day['temperature_2m'].max():.1f}°C"),
            ("🌡️ 最低気温", f"{df_day['temperature_2m'].min():.1f}°C"),
            ("🌧️ 最大降水確率", f"{df_day['precipitation_probability'].max():.0f}%"),
            ("💨 最大風速", f"{df_day['windspeed_10m'].max():.1f} m/s"),
            ("☔ 総降水量", f"{df_day['precipitation'].sum():.1f} mm"),
        ]
        for col, (label, val) in zip(m_cols, metrics):
            col.metric(label, val)

        # 1時間ごとテーブル
        display_df = df_day[["time", "temperature_2m", "apparent_temperature",
                               "precipitation_probability", "precipitation",
                               "windspeed_10m", "cloudcover", "weathercode"]].copy()
        display_df["時刻"] = display_df["time"].dt.strftime("%H:%M")
        display_df["天気"] = display_df["weathercode"].apply(get_weather_code_label)
        display_df["気温(°C)"] = display_df["temperature_2m"].round(1)
        display_df["体感気温(°C)"] = display_df["apparent_temperature"].round(1)
        display_df["降水確率(%)"] = display_df["precipitation_probability"].round(0).astype(int)
        display_df["降水量(mm)"] = display_df["precipitation"].round(1)
        display_df["風速(m/s)"] = display_df["windspeed_10m"].round(1)
        display_df["雲量(%)"] = display_df["cloudcover"].round(0).astype(int) if "cloudcover" in display_df else "-"

        st.dataframe(
            display_df[["時刻", "天気", "気温(°C)", "体感気温(°C)",
                          "降水確率(%)", "降水量(mm)", "風速(m/s)", "雲量(%)"]],
            use_container_width=True,
            hide_index=True,
        )

    today = ensemble_df["time"].dt.date.min()

    for i, tab in enumerate(tabs[:4]):
        with tab:
            target_date = today + timedelta(days=i) if i < 3 else None
            if i < 3:
                day_df = ensemble_df[ensemble_df["time"].dt.date == target_date]
                label = ["今日", "明日", "明後日"][i]
                render_day_detail(day_df, f"{label} ({target_date.strftime('%m/%d')})")
            else:
                # 3〜7日先
                start = today + timedelta(days=3)
                future_df = ensemble_df[ensemble_df["time"].dt.date >= start]
                for date in sorted(future_df["time"].dt.date.unique()):
                    day_df = future_df[future_df["time"].dt.date == date]
                    render_day_detail(day_df, date.strftime("%-m月%-d日"))
                    st.divider()

    # --- 全期間グラフ ---
    with tabs[4]:
        st.markdown("### 7日間 1時間ごと推移グラフ")

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "🌡️ 気温・体感気温 (°C)",
                "🌧️ 降水確率 (%) / 降水量 (mm)",
                "💨 風速 (m/s)",
                "☁️ 雲量 (%)",
            ),
            row_heights=[0.3, 0.25, 0.25, 0.2],
        )

        x = ensemble_df["time"]

        # 気温
        fig.add_trace(go.Scatter(x=x, y=ensemble_df["temperature_2m"],
            name="気温", line=dict(color="#ff6b35", width=2.5)), row=1, col=1)
        if "apparent_temperature" in ensemble_df:
            fig.add_trace(go.Scatter(x=x, y=ensemble_df["apparent_temperature"],
                name="体感気温", line=dict(color="#ffd166", width=1.5, dash="dot")), row=1, col=1)
        # 不確実性バンド
        if "temp_std" in ensemble_df:
            fig.add_trace(go.Scatter(
                x=pd.concat([x, x[::-1]]),
                y=pd.concat([
                    ensemble_df["temperature_2m"] + ensemble_df["temp_std"],
                    (ensemble_df["temperature_2m"] - ensemble_df["temp_std"])[::-1]
                ]),
                fill="toself", fillcolor="rgba(255,107,53,0.15)",
                line=dict(color="rgba(255,107,53,0)"), name="気温の不確実性範囲", showlegend=True
            ), row=1, col=1)

        # 降水
        fig.add_trace(go.Bar(x=x, y=ensemble_df["precipitation"],
            name="降水量(mm)", marker_color="rgba(0,119,182,0.6)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=ensemble_df["precipitation_probability"],
            name="降水確率(%)", line=dict(color="#0077b6", width=2),
            yaxis="y2"), row=2, col=1)

        # 風速
        fig.add_trace(go.Scatter(x=x, y=ensemble_df["windspeed_10m"],
            name="風速(m/s)", line=dict(color="#06d6a0", width=2),
            fill="tozeroy", fillcolor="rgba(6,214,160,0.15)"), row=3, col=1)
        # 警戒ライン
        fig.add_hline(y=5, line_dash="dash", line_color="orange",
                      annotation_text="注意(5m/s)", row=3, col=1)
        fig.add_hline(y=10, line_dash="dash", line_color="red",
                      annotation_text="警戒(10m/s)", row=3, col=1)

        # 雲量
        if "cloudcover" in ensemble_df:
            fig.add_trace(go.Scatter(x=x, y=ensemble_df["cloudcover"],
                name="雲量(%)", line=dict(color="#adb5bd", width=2),
                fill="tozeroy", fillcolor="rgba(173,181,189,0.2)"), row=4, col=1)

        fig.update_layout(
            height=820,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        fig.update_xaxes(gridcolor="#333", showgrid=True)
        fig.update_yaxes(gridcolor="#333", showgrid=True)

        st.plotly_chart(fig, use_container_width=True)

    # --- ゴルフ適性スコア ---
    st.divider()
    st.markdown("### 🏌️ ゴルフ適性スコア（1時間ごと）")

    def golf_score(row):
        score = 100
        # 降水確率
        score -= row.get("precipitation_probability", 0) * 0.5
        # 風速ペナルティ
        ws = row.get("windspeed_10m", 0)
        if ws > 10: score -= 30
        elif ws > 7: score -= 15
        elif ws > 5: score -= 5
        # 雲量（少し曇りが良い）
        cc = row.get("cloudcover", 50)
        if cc < 30: score += 5  # 快晴は暑い
        elif cc > 80: score -= 10
        # 気温
        t = row.get("temperature_2m", 20)
        if 15 <= t <= 28: score += 5
        elif t < 5 or t > 35: score -= 20
        return max(0, min(100, score))

    today_df = ensemble_df[ensemble_df["time"].dt.date == today].copy()
    today_df["ゴルフ適性"] = today_df.apply(golf_score, axis=1)
    today_df["時刻"] = today_df["time"].dt.strftime("%H:%M")
    today_df["スコア"] = today_df["ゴルフ適性"].round(0).astype(int)
    today_df["評価"] = today_df["スコア"].apply(
        lambda s: "🟢 最高" if s >= 80 else ("🟡 良好" if s >= 60 else ("🟠 注意" if s >= 40 else "🔴 困難"))
    )

    golf_fig = go.Figure(go.Bar(
        x=today_df["時刻"],
        y=today_df["スコア"],
        marker_color=today_df["スコア"].apply(
            lambda s: "#2dc653" if s >= 80 else ("#ffd166" if s >= 60 else ("#f4a261" if s >= 40 else "#e63946"))
        ),
        text=today_df["評価"],
        textposition="outside",
    ))
    golf_fig.update_layout(
        height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"), yaxis=dict(range=[0, 110]),
        xaxis_title="時刻", yaxis_title="ゴルフ適性スコア",
    )
    st.plotly_chart(golf_fig, use_container_width=True)

    st.caption(f"🔄 データ更新: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 使用モデル数: {sum(1 for s in model_status.values() if '✅' in s)}/{len(model_status)}")

else:
    st.markdown("""
    ---
    ### 使い方
    1. 上の検索欄にゴルフ場名を入力（日本語・英語どちらでも可）
    2. 「予報を取得」ボタンをクリック
    3. 世界8カ国の気象モデルをリアルタイムで集計し、最も精度の高い予報を表示します

    **対応している気象モデル：**
    | モデル | 提供機関 | 特徴 |
    |---|---|---|
    | ECMWF IFS | 欧州中期予報センター | 世界最高精度の中期予報 |
    | GFS | 米国NOAA | 全球高解像度 |
    | ICON | 独DWD | ヨーロッパ・アジア高精度 |
    | GEM | カナダ環境省 | 北米・太平洋域 |
    | JMA | 日本気象庁 | 日本域最高精度 |
    | Météo-France | フランス気象局 | 欧州精度トップクラス |
    | ACCESS-G | 豪BOM | 南半球高精度 |
    | ARPAE | イタリア | 地中海・欧州 |
    """)
