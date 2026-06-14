import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import time

st.set_page_config(
    page_title="⛳ ゴルフ場 精密天気予報",
    page_icon="⛳",
    layout="wide"
)

# ---- スタイル（スマホ対応） ----
st.markdown("""
<style>
body { font-family: 'Hiragino Sans', sans-serif; }

/* メトリクスカードをスマホで折り返す */
.summary-grid {
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;
}
.summary-card {
    background: #1e3a2f; border-radius: 10px; padding: 12px 16px;
    flex: 1 1 120px; min-width: 100px; text-align: center; color: white;
}
.summary-card .val { font-size: 22px; font-weight: bold; }
.summary-card .lbl { font-size: 11px; opacity: 0.75; margin-top: 2px; }

/* 的中率バナー */
.acc-banner {
    display: flex; flex-wrap: wrap; gap: 10px;
    background: #111; border-radius: 12px; padding: 14px; margin-bottom: 16px;
}
.acc-card {
    flex: 1 1 130px; text-align: center; border-radius: 10px; padding: 12px;
    color: white;
}
.acc-card .big { font-size: 32px; font-weight: bold; }
.acc-card .sub { font-size: 12px; opacity: 0.8; margin-top: 4px; }

/* テーブルの横スクロールをスマホで有効に */
[data-testid="stDataFrame"] { overflow-x: auto !important; }

/* サイドバー非表示時の余白調整 */
@media (max-width: 768px) {
    .block-container { padding-left: 12px !important; padding-right: 12px !important; }
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

# ---- 場所名・住所・ゴルフ場名をNominatimで検索 ----

@st.cache_data(ttl=3600, show_spinner=False)
def search_places(keyword: str) -> list:
    """Nominatimでキーワードを検索し、候補リストを返す（1〜2秒で完了）。
    ゴルフ場名・住所・市区町村名・何でも対応。"""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": keyword,
                "format": "jsonv2",
                "limit": 10,
                "addressdetails": 1,
                "countrycodes": "jp",
            },
            headers={"User-Agent": "golf-weather-app/1.0 (r.ishiyama73@gmail.com)"},
            timeout=8,
        )
        r.raise_for_status()
        items = r.json()
    except Exception:
        return []

    results = []
    seen_coords = set()
    for item in items:
        lat_r = round(float(item["lat"]), 3)
        lon_r = round(float(item["lon"]), 3)
        if (lat_r, lon_r) in seen_coords:
            continue
        seen_coords.add((lat_r, lon_r))

        addr_obj = item.get("address", {})
        addr_parts = [
            addr_obj.get("state", "") or addr_obj.get("province", ""),
            addr_obj.get("city", "") or addr_obj.get("county", "") or addr_obj.get("town", "") or addr_obj.get("village", ""),
        ]
        addr = " ".join(p for p in addr_parts if p)

        name = item.get("name") or item.get("display_name", "").split(",")[0]
        results.append({
            "name": name,
            "lat": float(item["lat"]),
            "lon": float(item["lon"]),
            "address": addr or item.get("display_name", "")[:40],
        })
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

# ---- 的中率・精度メトリクスを計算 ----
@st.cache_data(ttl=3600, show_spinner=False)
def compute_accuracy_metrics(lat, lon, models_to_use: tuple):
    """過去7日間の予報 vs 実測を比較して的中率を返す"""
    today = datetime.now().date()
    start = (today - timedelta(days=8)).isoformat()
    end   = (today - timedelta(days=1)).isoformat()

    actual_df = fetch_actual_weather(lat, lon, start, end)
    if actual_df is None or actual_df.empty:
        return None

    def fetch_hist(model_id):
        try:
            r = requests.get(
                "https://historical-forecast-api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "start_date": start, "end_date": end,
                    "hourly": ["temperature_2m", "precipitation", "precipitation_probability"],
                    "models": model_id,
                    "timezone": "auto",
                },
                timeout=15,
            )
            r.raise_for_status()
            df = pd.DataFrame(r.json()["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            return df
        except Exception:
            return None

    # 全モデルを並列取得
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_hist, mid): name for name, mid in dict(models_to_use).items()}
        model_results = {}
        for f in as_completed(futures):
            name = futures[f]
            df = f.result()
            if df is not None:
                model_results[name] = df

    if not model_results:
        return None

    # アンサンブル予報（単純平均）を作成
    pred_dfs = []
    for df in model_results.values():
        pred_dfs.append(df.set_index("time"))
    ensemble_pred = pd.concat(pred_dfs).groupby(level=0).mean()

    merged = actual_df.set_index("time").join(ensemble_pred, lsuffix="_act", rsuffix="_pred").dropna()
    if merged.empty:
        return None

    # 気温MAE → 的中率換算（許容誤差±2°C以内を「当たり」）
    temp_err = (merged["temperature_2m_act"] - merged["temperature_2m_pred"]).abs()
    temp_mae  = float(temp_err.mean())
    temp_hit  = float((temp_err <= 2.0).mean() * 100)  # ±2°C以内の割合

    # 降水的中率（予測50%以上 → 雨予報、実測0.5mm以上 → 雨）
    if "precipitation_probability_pred" in merged.columns and "precipitation_act" in merged.columns:
        pred_rain = merged["precipitation_probability_pred"] >= 50
        act_rain  = merged["precipitation_act"] >= 0.5
        precip_hit = float((pred_rain == act_rain).mean() * 100)
    else:
        precip_hit = None

    # 総合スコア（気温的中70% + 降水30%）
    if precip_hit is not None:
        overall = temp_hit * 0.7 + precip_hit * 0.3
    else:
        overall = temp_hit

    return {
        "temp_mae": temp_mae,
        "temp_hit": temp_hit,
        "precip_hit": precip_hit,
        "overall": overall,
        "days": 7,
        "hours": len(merged),
    }


# ---- 時間帯別ゴルフコメント生成 ----
def generate_hourly_comment(row) -> str:
    hour = row["time"].hour
    temp = row.get("temperature_2m", np.nan)
    feel = row.get("apparent_temperature", np.nan)
    precip_prob = row.get("precipitation_probability", 0) or 0
    precip = row.get("precipitation", 0) or 0
    wind = row.get("windspeed_10m", 0) or 0
    cloud = row.get("cloudcover", 50) or 50
    wcode = int(row.get("weathercode", 0) or 0)

    parts = []

    # 気温コメント
    if not np.isnan(temp):
        if temp >= 35:
            parts.append(f"気温{temp:.0f}°Cと猛暑。熱中症に厳重注意、こまめな水分補給を")
        elif temp >= 30:
            parts.append(f"気温{temp:.0f}°Cと真夏日。水分・塩分補給をこまめに")
        elif temp >= 25:
            parts.append(f"気温{temp:.0f}°Cと暑め。日焼け対策と水分補給を忘れずに")
        elif temp >= 18:
            parts.append(f"気温{temp:.0f}°Cと快適なプレー日和")
        elif temp >= 10:
            parts.append(f"気温{temp:.0f}°Cとやや肌寒め。ウィンドブレーカーが活躍")
        else:
            parts.append(f"気温{temp:.0f}°Cと寒い。重ね着で防寒対策を")

    # 体感気温が気温と大きく違う場合
    if not np.isnan(feel) and not np.isnan(temp) and abs(feel - temp) >= 4:
        if feel < temp:
            parts.append(f"風で体感は{feel:.0f}°Cまで下がる")
        else:
            parts.append(f"湿度で体感は{feel:.0f}°Cと蒸し暑く感じる")

    # 降水コメント
    if precip_prob >= 80:
        parts.append(f"降水確率{precip_prob:.0f}%と雨がほぼ確実。カッパ必携")
    elif precip_prob >= 60:
        parts.append(f"降水確率{precip_prob:.0f}%。傘・カッパを必ず用意して")
    elif precip_prob >= 40:
        parts.append(f"降水確率{precip_prob:.0f}%。折りたたみ傘を念のため")
    elif precip_prob >= 20:
        parts.append(f"降水確率{precip_prob:.0f}%とにわか雨の可能性あり")

    if precip >= 5:
        parts.append(f"1時間に{precip:.1f}mmの強雨予想")
    elif precip >= 1:
        parts.append(f"降水量{precip:.1f}mm予想")

    # 風コメント
    if wind >= 12:
        parts.append(f"風速{wind:.1f}m/sの強風。クラブ選択と弾道に要注意")
    elif wind >= 8:
        parts.append(f"風速{wind:.1f}m/sのやや強い風。アゲインスト・フォローを意識して")
    elif wind >= 5:
        parts.append(f"風速{wind:.1f}m/sの風あり。ショートゲームへの影響を考慮")
    elif wind >= 3:
        parts.append(f"微風({wind:.1f}m/s)でプレーしやすい")
    else:
        parts.append("ほぼ無風で狙いどおりのショットが期待できる")

    # 雲量コメント
    if cloud <= 20:
        parts.append("快晴で視界良好")
    elif cloud <= 50:
        parts.append("晴れ間が広がり気持ちの良いラウンド")
    elif cloud <= 80:
        parts.append("曇りがちだが直射日光が遮られ逆に快適")
    else:
        parts.append("厚い雲に覆われ薄暗め")

    return "。".join(parts) + "。"


# ---- 複数モデルを加重アンサンブル集計（並列取得） ----
def build_ensemble_df(lat, lon, models_to_use, use_weighted=True):
    all_dfs = []
    model_status = {}

    # 精度重み計算と全モデル取得を並列実行
    with st.spinner("気象モデルを並列取得中..."):
        with ThreadPoolExecutor(max_workers=10) as ex:
            # 重み計算
            weight_future = ex.submit(
                compute_model_weights, lat, lon, tuple(models_to_use.items())
            ) if use_weighted else None

            # 全モデルを同時リクエスト
            forecast_futures = {
                ex.submit(fetch_model_forecast, lat, lon, model_id): name
                for name, model_id in models_to_use.items()
            }

            weights = weight_future.result() if weight_future else {n: 1.0 for n in models_to_use}

            for future in as_completed(forecast_futures):
                name = forecast_futures[future]
                data = future.result()
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

col1, col2 = st.columns([3, 1])
def _do_search():
    kw = st.session_state.keyword_input
    if kw and len(kw) >= 2:
        st.session_state.candidates = search_places(kw)

with col1:
    keyword = st.text_input(
        "ゴルフ場名・住所・市区町村名を入力",
        placeholder="例: オーク・ヒルズCC、成田市、千葉県富里市十倉、Pebble Beach",
        label_visibility="collapsed",
        key="keyword_input",
        on_change=_do_search,
    )
with col2:
    search_btn = st.button("🔍 候補を検索", type="secondary", use_container_width=True)

# ボタン押下でも検索
if search_btn and keyword and len(keyword) >= 2:
    with st.spinner(f"「{keyword}」を検索中..."):
        st.session_state.candidates = search_places(keyword)

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

elif search_btn and keyword and len(keyword) >= 2:
    st.warning("見つかりませんでした。別のキーワードや住所で試してください。")

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

    # --- 的中率 & アンサンブル取得を並列実行 ---
    with ThreadPoolExecutor(max_workers=2) as ex:
        acc_future = ex.submit(compute_accuracy_metrics, lat, lon, tuple(selected_models.items()))
        # アンサンブルはメインスレッドでStreamlit UIを使うため逐次実行
    ensemble_df, model_status, weights = build_ensemble_df(lat, lon, selected_models, use_weighted=True)
    acc = acc_future.result()

    if ensemble_df is None:
        st.error("天気データの取得に失敗しました。しばらく待ってから再試行してください。")
        st.stop()

    # --- 的中率バナー ---
    if acc:
        st.markdown("### 📊 予報的中率（過去7日間の実績）")
        score_color = "#2dc653" if acc["overall"] >= 80 else ("#ffd166" if acc["overall"] >= 65 else "#e63946")
        precip_card = f'<div class="acc-card" style="background:#1a3a5c"><div class="big">{acc["precip_hit"]:.1f}%</div><div class="sub">🌧️ 降水的中率</div></div>' if acc["precip_hit"] is not None else ""
        st.markdown(f"""
<div class="acc-banner">
  <div class="acc-card" style="background:{score_color};color:#000;flex:1 1 160px">
    <div class="big">{acc["overall"]:.1f}%</div>
    <div class="sub" style="color:#000">総合的中率</div>
  </div>
  <div class="acc-card" style="background:#1a3a2f">
    <div class="big">{acc["temp_hit"]:.1f}%</div>
    <div class="sub">🌡️ 気温的中率（±2°C以内）</div>
    <div class="sub">平均誤差 ±{acc["temp_mae"]:.1f}°C</div>
  </div>
  {precip_card}
  <div class="acc-card" style="background:#2a2a2a">
    <div class="big">{acc["days"]}日間</div>
    <div class="sub">📅 検証期間（計{acc["hours"]}h）</div>
  </div>
</div>
""", unsafe_allow_html=True)
        st.divider()

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
            st.plotly_chart(wfig, use_container_width=True, config={"staticPlot": True, "displayModeBar": False})
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
        cards_html = "".join([
            f'<div class="summary-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
            for l, v in [
                ("最高気温", f"{df_day['temperature_2m'].max():.1f}°C"),
                ("最低気温", f"{df_day['temperature_2m'].min():.1f}°C"),
                ("最大降水確率", f"{df_day['precipitation_probability'].max():.0f}%"),
                ("最大風速", f"{df_day['windspeed_10m'].max():.1f} m/s"),
                ("総降水量", f"{df_day['precipitation'].sum():.1f} mm"),
            ]
        ])
        st.markdown(f'<div class="summary-grid">{cards_html}</div>', unsafe_allow_html=True)

        # 1時間ごとテーブル
        def to_int_safe(s):
            return s.fillna(0).round(0).astype(int)

        cols_needed = ["time", "temperature_2m", "apparent_temperature",
                       "precipitation_probability", "precipitation",
                       "windspeed_10m", "cloudcover", "weathercode"]
        display_df = df_day[[c for c in cols_needed if c in df_day.columns]].copy()
        for c in cols_needed:
            if c not in display_df.columns:
                display_df[c] = np.nan

        display_df["時刻"] = display_df["time"].dt.strftime("%H:%M")
        display_df["天気"] = display_df["weathercode"].fillna(0).apply(get_weather_code_label)
        display_df["気温(°C)"] = display_df["temperature_2m"].round(1)
        display_df["降水確率(%)"] = to_int_safe(display_df["precipitation_probability"])
        display_df["降水量(mm)"] = display_df["precipitation"].fillna(0).round(1)
        display_df["風速(m/s)"] = display_df["windspeed_10m"].round(1)
        display_df["コメント"] = display_df.apply(generate_hourly_comment, axis=1)

        cols_show = ["時刻", "天気", "気温(°C)", "降水確率(%)", "降水量(mm)", "風速(m/s)", "コメント"]
        rows_html = ""
        for _, r in display_df[cols_show].iterrows():
            cells = []
            for i, (c, v) in enumerate(r.items()):
                if i == 0:  # 時刻列を横スクロール時に固定
                    cells.append(f'<td style="white-space:nowrap;padding:6px 10px;border-bottom:1px solid #ddd;position:sticky;left:0;background:inherit;z-index:1;font-weight:bold">{v}</td>')
                elif c != "コメント":
                    cells.append(f'<td style="white-space:nowrap;padding:6px 10px;border-bottom:1px solid #ddd">{v}</td>')
                else:
                    cells.append(f'<td style="padding:6px 10px;border-bottom:1px solid #ddd;min-width:200px">{v}</td>')
            rows_html += "<tr>" + "".join(cells) + "</tr>"

        header_cells = []
        for i, c in enumerate(cols_show):
            if i == 0:  # 時刻ヘッダーも固定
                header_cells.append(f'<th style="padding:6px 10px;background:#1e3a2f;color:#ffffff;text-align:left;white-space:nowrap;position:sticky;left:0;z-index:3">{c}</th>')
            else:
                header_cells.append(f'<th style="padding:6px 10px;background:#1e3a2f;color:#ffffff;text-align:left;white-space:nowrap">{c}</th>')
        header_html = "".join(header_cells)

        st.markdown(f"""
<div style="overflow-x:auto">
<table style="width:100%;border-collapse:collapse;font-size:13px;color:inherit">
<thead><tr>{header_html}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

        # ゴルフ適性スコア（天気スコアのみ）
        st.markdown("#### 🏌️ ゴルフ適性スコア")
        score_df = display_df.copy()

        def calc_score(row):
            s = 100
            pp = row.get("降水確率(%)", 0) or 0
            pr = row.get("降水量(mm)", 0) or 0
            ws = row.get("風速(m/s)", 0) or 0
            t  = row.get("気温(°C)", 20) or 20
            if pp >= 80:    s -= 60
            elif pp >= 60:  s -= 40
            elif pp >= 40:  s -= 20
            elif pp >= 20:  s -= 10
            if pr >= 5:     s -= 30   # 強雨
            elif pr >= 2:   s -= 20   # 雨
            elif pr >= 1:   s -= 8    # 小雨
            elif pr >= 0.3: s -= 3    # 霧雨程度・ほぼ問題なし
            if ws >= 12:    s -= 45
            elif ws >= 8:   s -= 30
            elif ws >= 6:   s -= 15
            elif ws >= 4:   s -= 8
            if 16 <= t <= 26:      s += 5
            elif t < 5 or t > 35:  s -= 25
            elif t < 10 or t > 32: s -= 12
            return max(0, min(100, s))

        score_df["スコア"] = score_df.apply(calc_score, axis=1).astype(int)

        def bar_colors(vals):
            return ["#2dc653" if v >= 80 else "#ffd166" if v >= 60 else "#f4a261" if v >= 40 else "#e63946" for v in vals]

        gfig = go.Figure(go.Bar(
            x=score_df["時刻"],
            y=score_df["スコア"],
            marker_color=bar_colors(score_df["スコア"]),
            text=score_df["スコア"],
            textposition="outside",
        ))
        gfig.update_layout(
            height=320, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"), showlegend=False,
            yaxis=dict(range=[0, 120]),
            xaxis=dict(
                tickmode="array",
                tickvals=score_df["時刻"].tolist(),
                ticktext=score_df["時刻"].tolist(),
                tickangle=-45,
            ),
            margin=dict(t=20, b=60),
            bargap=0.15,
        )
        gfig.update_xaxes(gridcolor="#333")
        gfig.update_yaxes(gridcolor="#333")
        st.plotly_chart(gfig, use_container_width=True, config={"staticPlot": True, "displayModeBar": False})
        st.caption("🟢 80点以上: 最高　🟡 60〜79点: 良好　🟠 40〜59点: 注意　🔴 39点以下: 困難")

    today = ensemble_df["time"].dt.date.min()

    for i, tab in enumerate(tabs[:4]):
        with tab:
            target_date = today + timedelta(days=i) if i < 3 else None
            if i < 3:
                day_df = ensemble_df[ensemble_df["time"].dt.date == target_date]
                WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]
                label = ["今日", "明日", "明後日"][i]
                dow = WEEKDAYS[target_date.weekday()]
                render_day_detail(day_df, f"{label} ({target_date.strftime('%m/%d')}・{dow})")
            else:
                # 3〜7日先
                WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]
                start = today + timedelta(days=3)
                future_df = ensemble_df[ensemble_df["time"].dt.date >= start]
                for date in sorted(future_df["time"].dt.date.unique()):
                    day_df = future_df[future_df["time"].dt.date == date]
                    dow = WEEKDAYS[date.weekday()]
                    render_day_detail(day_df, f"{date.month}月{date.day}日（{dow}）")
                    st.divider()

    # --- 全期間グラフ ---
    with tabs[4]:
        st.markdown("### 7日間 1時間ごと推移グラフ")

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "🌡️ 気温 (°C)",
                "🌧️ 降水確率 (%) / 降水量 (mm)",
                "💨 風速 (m/s)",
            ),
            row_heights=[0.35, 0.35, 0.30],
        )

        x = ensemble_df["time"]

        # 気温
        fig.add_trace(go.Scatter(x=x, y=ensemble_df["temperature_2m"],
            name="気温", line=dict(color="#ff6b35", width=2.5)), row=1, col=1)
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
        fig.add_hline(y=5, line_dash="dash", line_color="orange",
                      annotation_text="注意(5m/s)", row=3, col=1)
        fig.add_hline(y=10, line_dash="dash", line_color="red",
                      annotation_text="警戒(10m/s)", row=3, col=1)

        fig.update_layout(
            height=680,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        fig.update_xaxes(gridcolor="#333", showgrid=True)
        fig.update_yaxes(gridcolor="#333", showgrid=True)

        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, "displayModeBar": False})

    st.caption(f"🔄 データ更新: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 使用モデル数: {sum(1 for s in model_status.values() if '✅' in s)}/{len(model_status)}")

else:
    st.markdown("""
    ---
    ### 使い方
    1. 上の検索欄に **ゴルフ場名・住所・市区町村名** など何でも入力する（例: **オーク・ヒルズCC**、**成田市**、**千葉県富里市**、**Pebble Beach**）
    2. 「候補を検索」ボタンを押すと候補がプルダウンで表示される
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
