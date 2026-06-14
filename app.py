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

# ---- ゴルフ場名候補をNominatimで検索 ----
def _nominatim_search(query: str) -> list:
    """Nominatim に1クエリ投げてゴルフ場リストを返す"""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 30,
                "extratags": 1,
                "addressdetails": 1,
            },
            headers={"User-Agent": "golf-weather-app/1.0 (r.ishiyama73@gmail.com)"},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def search_golf_courses(keyword: str):
    """キーワードに部分一致するゴルフ場をOSMから検索し、候補リストを返す。
    日英両方のクエリを投げて結果をマージする。"""

    # 日英複数クエリ（「ゴルフ場」「ゴルフクラブ」「golf course」「golf club」を付与）
    queries = [
        f"{keyword} ゴルフ場",
        f"{keyword} ゴルフクラブ",
        f"{keyword} golf course",
        f"{keyword} golf club",
        keyword,
    ]

    raw = []
    seen_queries = set()
    for q in queries:
        if q in seen_queries:
            continue
        seen_queries.add(q)
        raw.extend(_nominatim_search(q))
        time.sleep(0.3)  # Nominatim利用規約：1req/秒以下

    results = []
    seen_names = set()
    seen_coords = set()
    for item in raw:
        # golf_courseタイプ、またはdisplay_nameに"golf"を含むものに絞り込み
        item_type = item.get("type", "")
        display = item.get("display_name", "").lower()
        et = item.get("extratags") or {}
        is_golf = (
            item_type in ("golf_course", "golf")
            or et.get("leisure") == "golf_course"
            or ("golf" in display and item_type in ("", "yes", "leisure", "club"))
        )
        if not is_golf:
            continue

        name = item.get("name") or display.split(",")[0]
        if not name or name in seen_names:
            continue

        # 座標が近すぎる重複を除外（同一施設の別ノード）
        lat_r = round(float(item["lat"]), 3)
        lon_r = round(float(item["lon"]), 3)
        coord_key = (lat_r, lon_r)
        if coord_key in seen_coords:
            continue

        seen_names.add(name)
        seen_coords.add(coord_key)

        addr_obj = item.get("address", {})
        addr_parts = [
            addr_obj.get("country", ""),
            addr_obj.get("state", "") or addr_obj.get("province", ""),
            addr_obj.get("city", "") or addr_obj.get("county", "") or addr_obj.get("town", ""),
        ]
        addr = " ".join(p for p in addr_parts if p) or "住所不明"
        results.append({
            "name": name,
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "address": addr,
        })

    results.sort(key=lambda x: x["name"])
    return results

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

# ---- 過去の実測値を取得（Open-Meteo Historical API）----
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_actual_weather(lat, lon, start_date: str, end_date: str):
    """指定期間の実測気象データを取得する"""
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": start_date, "end_date": end_date,
                "hourly": ["temperature_2m", "precipitation", "windspeed_10m"],
                "timezone": "auto", "wind_speed_unit": "ms",
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception:
        return None

# ---- 過去3日間の各モデル予報 vs 実測でMAEを計算し重みを返す ----
@st.cache_data(ttl=3600, show_spinner=False)
def compute_model_weights(lat, lon, models_to_use: tuple):
    """過去3日間の予報誤差（MAE）を計算し、精度の高いモデルに大きな重みを付ける"""
    today = datetime.now().date()
    start = (today - timedelta(days=4)).isoformat()
    end   = (today - timedelta(days=1)).isoformat()

    actual_df = fetch_actual_weather(lat, lon, start, end)
    if actual_df is None or actual_df.empty:
        # 実測が取れない場合は均等重み
        return {name: 1.0 for name in dict(models_to_use)}

    model_mae = {}
    for name, model_id in dict(models_to_use).items():
        try:
            r = requests.get(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "start_date": start, "end_date": end,
                    "hourly": ["temperature_2m", "precipitation", "windspeed_10m"],
                    "models": model_id,
                    "timezone": "auto", "wind_speed_unit": "ms",
                },
                timeout=15,
            )
            r.raise_for_status()
            pred_df = pd.DataFrame(r.json()["hourly"])
            pred_df["time"] = pd.to_datetime(pred_df["time"])

            merged = actual_df.merge(pred_df, on="time", suffixes=("_act", "_pred"))
            if merged.empty:
                continue
            mae = (merged["temperature_2m_act"] - merged["temperature_2m_pred"]).abs().mean()
            model_mae[name] = float(mae)
        except Exception:
            pass
        time.sleep(0.05)

    if not model_mae:
        return {name: 1.0 for name in dict(models_to_use)}

    # MAEが小さいほど重みを大きく（逆数比例）
    min_mae = min(model_mae.values()) + 0.01
    weights = {}
    for name in dict(models_to_use):
        mae = model_mae.get(name, max(model_mae.values()) * 1.5)
        weights[name] = min_mae / (mae + 0.01)

    # 正規化
    total = sum(weights.values())
    return {k: v / total * len(weights) for k, v in weights.items()}

# ---- 複数モデルを加重アンサンブル集計 ----
def build_ensemble_df(lat, lon, models_to_use, use_weighted=True):
    all_dfs = []
    model_status = {}

    # 精度ベースの重み計算（バックグラウンドで並行実行）
    weights = {}
    if use_weighted:
        with st.spinner("過去実績から各モデルの精度を計算中..."):
            weights = compute_model_weights(lat, lon, tuple(models_to_use.items()))

    progress = st.progress(0, text="気象モデルを取得中...")
    for i, (name, model_id) in enumerate(models_to_use.items()):
        progress.progress((i + 1) / len(models_to_use), text=f"取得中: {name}")
        data = fetch_model_forecast(lat, lon, model_id)
        if data and "hourly" in data:
            df = pd.DataFrame(data["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            df["model"] = name
            df["weight"] = weights.get(name, 1.0)
            all_dfs.append(df)
            w = weights.get(name, 1.0)
            model_status[name] = f"✅ (精度重み: {w:.2f})"
        else:
            model_status[name] = "❌ (取得失敗)"
        time.sleep(0.1)

    progress.empty()

    if not all_dfs:
        return None, model_status, {}

    numeric_cols = [
        "temperature_2m", "precipitation_probability", "precipitation",
        "windspeed_10m", "winddirection_10m", "apparent_temperature",
        "cloudcover", "uv_index", "visibility",
    ]

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["weight"] = combined["weight"].fillna(1.0)

    # 時刻ごとに加重平均
    times = combined["time"].unique()
    rows = []
    for t in times:
        grp = combined[combined["time"] == t]
        w = grp["weight"].values
        row = {"time": t}
        for col in numeric_cols:
            if col not in grp.columns:
                continue
            vals = grp[col].values.astype(float)
            valid = ~np.isnan(vals)
            if valid.any():
                row[col] = float(np.average(vals[valid], weights=w[valid]))
            else:
                row[col] = np.nan
        # weathercodeは最多数決
        if "weathercode" in grp.columns:
            wc = grp["weathercode"].dropna()
            row["weathercode"] = float(wc.mode().iloc[0]) if not wc.empty else 0.0
        rows.append(row)

    ensemble_df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    std_df = combined.groupby("time")[["temperature_2m", "precipitation"]].std().reset_index()
    std_df.columns = ["time", "temp_std", "precip_std"]
    ensemble_df = ensemble_df.merge(std_df, on="time", how="left")

    return ensemble_df, model_status, weights

# ---- メイン UI ----
st.title("⛳ ゴルフ場 精密天気予報")
st.caption("世界8カ国の気象モデルをアンサンブル集計 — 統計的に最も精度の高い予報を提供")

# セッション初期化
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "selected_course" not in st.session_state:
    st.session_state.selected_course = None

col1, col2 = st.columns([3, 1])
with col1:
    keyword = st.text_input(
        "ゴルフ場名を入力（2文字以上で候補が出ます）",
        placeholder="例: オーク、太平洋、Pebble Beach",
        label_visibility="collapsed",
        key="keyword_input",
    )
with col2:
    search_btn = st.button("🔍 候補を検索", type="secondary", use_container_width=True)

# キーワードが2文字以上になったら自動検索
if keyword and len(keyword) >= 2:
    if search_btn or (
        "last_keyword" not in st.session_state
        or st.session_state.last_keyword != keyword
    ):
        with st.spinner(f"「{keyword}」に一致するゴルフ場を検索中..."):
            st.session_state.candidates = search_golf_courses(keyword)
            st.session_state.last_keyword = keyword
            st.session_state.selected_course = None

# 候補プルダウン表示
lat, lon, address, course_name = None, None, None, None
forecast_btn = False

if st.session_state.candidates:
    options = [f"{c['name']}　（{c['address']}）" for c in st.session_state.candidates]
    chosen = st.selectbox(
        f"候補 {len(options)} 件 — 選んでください",
        options,
        index=0,
        key="course_select",
    )
    chosen_idx = options.index(chosen)
    chosen_data = st.session_state.candidates[chosen_idx]
    lat = chosen_data["lat"]
    lon = chosen_data["lon"]
    course_name = chosen_data["name"]
    address = chosen_data["address"]

    forecast_btn = st.button("⛅ この場所の予報を取得", type="primary")

elif keyword and len(keyword) >= 2 and "last_keyword" in st.session_state:
    st.warning("ゴルフ場が見つかりませんでした。別のキーワードをお試しください（例: 英語名、省略なしの正式名称）。")

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

if forecast_btn and lat:
    st.success(f"📍 **{course_name}**  |  {address}  |  緯度 {lat:.4f} / 経度 {lon:.4f}")

    # --- アンサンブル取得（加重） ---
    ensemble_df, model_status, weights = build_ensemble_df(lat, lon, selected_models, use_weighted=True)

    if ensemble_df is None:
        st.error("天気データの取得に失敗しました。しばらく待ってから再試行してください。")
        st.stop()

    # モデル取得状況 + 精度重みグラフ
    with st.expander("📡 モデル取得状況 & 精度重み（過去3日間の実績比較）"):
        for name, status in model_status.items():
            st.write(f"{status} {name}")

        if weights and any(v != 1.0 for v in weights.values()):
            st.markdown("##### 精度重み（高いほど今回の予報に大きく反映）")
            sorted_w = sorted(weights.items(), key=lambda x: -x[1])
            w_names = [n.split("(")[0].strip() for n, _ in sorted_w]
            w_vals  = [v for _, v in sorted_w]
            wfig = go.Figure(go.Bar(
                x=w_vals, y=w_names, orientation="h",
                marker_color=["#2dc653" if v >= 1.0 else "#f4a261" for v in w_vals],
                text=[f"{v:.2f}" for v in w_vals], textposition="outside",
            ))
            wfig.update_layout(
                height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="white", size=11),
                xaxis=dict(range=[0, max(w_vals) * 1.3]),
                margin=dict(l=10, r=40, t=10, b=10),
            )
            st.plotly_chart(wfig, use_container_width=True)
            st.caption("※ 過去3日間の気温予報 vs 実測値（MAE）をもとに自動計算")

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
        def to_int_safe(s):
            return s.fillna(0).round(0).astype(int)

        display_df["時刻"] = display_df["time"].dt.strftime("%H:%M")
        display_df["天気"] = display_df["weathercode"].fillna(0).apply(get_weather_code_label)
        display_df["気温(°C)"] = display_df["temperature_2m"].round(1)
        display_df["体感気温(°C)"] = display_df["apparent_temperature"].round(1)
        display_df["降水確率(%)"] = to_int_safe(display_df["precipitation_probability"])
        display_df["降水量(mm)"] = display_df["precipitation"].fillna(0).round(1)
        display_df["風速(m/s)"] = display_df["windspeed_10m"].round(1)
        display_df["雲量(%)"] = to_int_safe(display_df["cloudcover"]) if "cloudcover" in display_df.columns else "-"

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
    today_df["スコア"] = today_df["ゴルフ適性"].fillna(0).round(0).astype(int)
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
    1. 上の検索欄にゴルフ場名を **2文字以上** 入力する（例: **オーク**、**太平洋**、**Pebble**）
    2. 「候補を検索」ボタンを押すと一致するゴルフ場がプルダウンで表示される
    3. リストから行きたいゴルフ場を選んで「この場所の予報を取得」をクリック
    4. 世界8カ国の気象モデルをリアルタイムで集計し、最も精度の高い予報を表示します

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
