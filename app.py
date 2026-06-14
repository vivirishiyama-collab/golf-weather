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

    # 時間帯の挨拶
    if hour < 6:
        parts.append("夜明け前のスタート")
    elif hour < 9:
        parts.append("朝の爽やかな時間帯")
    elif hour < 12:
        parts.append("午前のプレーに最適な時間")
    elif hour < 14:
        parts.append("日差しが最も強い昼時")
    elif hour < 17:
        parts.append("午後の時間帯")
    else:
        parts.append("夕方のラウンド")

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
        st.markdown("### 📊 このゴルフ場における予報的中率（過去7日間実績）")
        a1, a2, a3, a4 = st.columns(4)
        score_color = "#2dc653" if acc["overall"] >= 80 else ("#ffd166" if acc["overall"] >= 65 else "#e63946")
        a1.markdown(f"""
        <div style='background:{score_color};border-radius:12px;padding:16px;text-align:center;color:#000'>
        <div style='font-size:13px;font-weight:bold'>総合的中率</div>
        <div style='font-size:38px;font-weight:bold'>{acc["overall"]:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        a2.metric("🌡️ 気温的中率", f"{acc['temp_hit']:.1f}%",
                  help="予報と実測の差が±2°C以内だった割合")
        a2.caption(f"平均誤差: ±{acc['temp_mae']:.1f}°C")
        if acc["precip_hit"] is not None:
            a3.metric("🌧️ 降水的中率", f"{acc['precip_hit']:.1f}%",
                      help="雨予報(確率50%以上)と実際の降雨(0.5mm以上)の一致率")
        a4.metric("📅 検証期間", f"過去{acc['days']}日間",
                  help=f"計{acc['hours']}時間分のデータで検証")
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
        display_df["体感気温(°C)"] = display_df["apparent_temperature"].round(1)
        display_df["降水確率(%)"] = to_int_safe(display_df["precipitation_probability"])
        display_df["降水量(mm)"] = display_df["precipitation"].fillna(0).round(1)
        display_df["風速(m/s)"] = display_df["windspeed_10m"].round(1)
        display_df["雲量(%)"] = to_int_safe(display_df["cloudcover"])
        display_df["コメント"] = display_df.apply(generate_hourly_comment, axis=1)

        st.dataframe(
            display_df[["時刻", "天気", "気温(°C)", "体感気温(°C)",
                          "降水確率(%)", "降水量(mm)", "風速(m/s)", "雲量(%)", "コメント"]],
            use_container_width=True,
            hide_index=True,
            column_config={"コメント": st.column_config.TextColumn(width="large")},
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
