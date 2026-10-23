import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import forecast_core as core

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

/* 検証バナー */
.acc-banner {
    display: flex; flex-wrap: wrap; gap: 10px;
    background: #111; border-radius: 12px; padding: 14px; margin-bottom: 16px;
}
.acc-card {
    flex: 1 1 130px; text-align: center; border-radius: 10px; padding: 12px;
    color: white;
}
.acc-card .big { font-size: 28px; font-weight: bold; }
.acc-card .sub { font-size: 12px; opacity: 0.8; margin-top: 4px; }

/* テーブルの横スクロールをスマホで有効に */
[data-testid="stDataFrame"] { overflow-x: auto !important; }


/* サイドバー非表示時の余白調整 */
@media (max-width: 768px) {
    .block-container { padding-left: 12px !important; padding-right: 12px !important; }
}
</style>
""", unsafe_allow_html=True)

WEATHER_MODELS = core.WEATHER_MODELS
MODEL_NAME = {v: k for k, v in WEATHER_MODELS.items()}

WEATHER_CODE = {
    0: "☀️ 快晴", 1: "🌤️ 晴れ", 2: "⛅ 曇りがち", 3: "☁️ 曇り",
    45: "🌫️ 霧", 48: "🌫️ 霧氷", 51: "🌦️ 霧雨(弱)", 53: "🌦️ 霧雨",
    55: "🌧️ 霧雨(強)", 61: "🌧️ 小雨", 63: "🌧️ 雨", 65: "🌧️ 大雨",
    71: "🌨️ 小雪", 73: "🌨️ 雪", 75: "❄️ 大雪", 77: "🌨️ 霙",
    80: "🌦️ にわか雨", 81: "🌦️ にわか雨", 82: "⛈️ にわか雨(強)",
    85: "🌨️ にわか雪", 86: "❄️ にわか大雪",
    95: "⛈️ 雷雨", 96: "⛈️ 雷雨+雹", 99: "⛈️ 激しい雷雨",
}

def get_weather_code_label(code):
    return WEATHER_CODE.get(int(code) if not np.isnan(code) else 0, f"コード{int(code)}")

# ---- 場所名・住所・ゴルフ場名を検索 ----

@st.cache_data(ttl=3600, show_spinner=False)
def _search_places_cached(keyword: str) -> list:
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": keyword,
            "count": 10,
            "language": "ja",
            "format": "json",
        },
        timeout=10,
    )
    r.raise_for_status()
    items = r.json().get("results") or []

    results = []
    seen_coords = set()
    for item in items:
        lat_r = round(float(item["latitude"]), 3)
        lon_r = round(float(item["longitude"]), 3)
        if (lat_r, lon_r) in seen_coords:
            continue
        seen_coords.add((lat_r, lon_r))

        addr_parts = [
            item.get("admin1", ""),
            item.get("admin2", "") or item.get("admin3", ""),
        ]
        addr = " ".join(p for p in addr_parts if p)

        results.append({
            "name": item.get("name", keyword),
            "lat": float(item["latitude"]),
            "lon": float(item["longitude"]),
            "address": addr or item.get("country", ""),
        })
    return results

def search_places(keyword: str) -> list:
    import re

    # 0. 緯度経度・GoogleマップのURLならそのまま使う（ゴルフ場の正確な位置で予報できる）
    coords = core.parse_coords(keyword)
    if coords:
        return [{"name": "指定した地点", "lat": coords[0], "lon": coords[1],
                 "address": f"緯度 {coords[0]:.4f} / 経度 {coords[1]:.4f}"}]

    def _try(kw):
        try:
            return _search_places_cached(kw)
        except Exception:
            return []

    # 1. そのまま検索
    results = _try(keyword)
    if results:
        return results

    # 2. 都道府県プレフィックスを除去（例：「東京都新宿区」→「新宿区」）
    stripped = re.sub(r'^.+?[都道府県]', '', keyword).strip()
    if stripped and stripped != keyword:
        results = _try(stripped)
        if results:
            return results

    # 3. 末尾の区・市・町・村・郡を除去（例：「新宿区」→「新宿」）
    trimmed = re.sub(r'[区市町村郡]$', '', stripped or keyword).strip()
    if trimmed and trimmed != (stripped or keyword):
        results = _try(trimmed)

    return results

# ---- データ取得（キャッシュ付き） ----
@st.cache_data(ttl=1800, show_spinner=False)
def cached_forecasts(lat, lon, model_ids: tuple):
    return core.fetch_forecasts(lat, lon, model_ids)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_ensemble(lat, lon):
    try:
        return core.fetch_ensemble_precip(lat, lon)
    except Exception:
        return None, None

@st.cache_data(ttl=1800, show_spinner=False)
def cached_jma_pop(lat, lon):
    try:
        return core.fetch_jma_official_pop(lat, lon)
    except Exception:
        return [], None

@st.cache_data(ttl=3600, show_spinner=False)
def cached_verification(lat, lon, model_ids: tuple):
    """最寄りアメダスの過去7日の実測と各モデルの予報を比べる"""
    try:
        station = core.find_amedas_station(lat, lon)
    except Exception:
        return None, {}
    if not station:
        return None, {}
    today = core.now_local(9 * 3600).date()
    start, end = today - timedelta(days=7), today - timedelta(days=1)
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_obs = ex.submit(core.fetch_amedas_obs, station["code"], start, end)
            f_past = ex.submit(core.fetch_past_forecasts, station["lat"], station["lon"],
                               station["alt"], list(model_ids), start, end)
            obs, past = f_obs.result(), f_past.result()
        return station, core.verify_models(obs, past)
    except Exception:
        return station, {}

def load_forecast(lat, lon, model_ids):
    jp = core.is_japan(lat, lon)
    with st.spinner("気象モデル・アンサンブル・アメダス実測を並列取得中..."):
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_fc = ex.submit(cached_forecasts, lat, lon, model_ids)
            f_ens = ex.submit(cached_ensemble, lat, lon)
            f_pop = ex.submit(cached_jma_pop, lat, lon) if jp else None
            f_ver = ex.submit(cached_verification, lat, lon, model_ids) if jp else None
            fc, utc_off, elevation = f_fc.result()
            ens, ens_groups = f_ens.result()
            jma_pop, jma_area = f_pop.result() if f_pop else ([], None)
            station, verif = f_ver.result() if f_ver else (None, {})
    if fc is None or fc.empty:
        return None
    now = core.now_local(utc_off)
    blend = core.build_blend(fc, now, jp, verif, ens, ens_groups, jma_pop)
    return {
        "df": blend, "now": now, "jp": jp, "elevation": elevation,
        "station": station, "verif": verif, "jma_area": jma_area,
        "models_ok": sorted(fc["model"].unique()), "ens_ok": ens is not None and not ens.empty,
        "jma_pop": jma_pop,
    }


# ---- 時間帯別ゴルフコメント生成 ----
def generate_hourly_comment(row) -> str:
    temp = row.get("temperature_2m", np.nan)
    feel = row.get("apparent_temperature", np.nan)
    prob = (row.get("rain_prob", 0) or 0) * 100
    amount = row.get("rain_amount", 0) or 0
    wind = row.get("windspeed_10m", 0) or 0
    gust = row.get("windgusts_10m", np.nan)
    cloud = row.get("cloudcover", 50)
    cloud = 50 if cloud is None or np.isnan(cloud) else cloud

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

    # 雨コメント（確率と、降った場合の強さ）
    strength = "強い雨" if amount >= 4 else ("雨" if amount >= 1 else "弱い雨")
    if prob >= 80:
        parts.append(f"雨の可能性{prob:.0f}%とほぼ確実（降れば{amount:.1f}mm/hの{strength}）。カッパ必携")
    elif prob >= 60:
        parts.append(f"雨の可能性{prob:.0f}%（降れば{amount:.1f}mm/hの{strength}）。カッパを必ず用意して")
    elif prob >= 40:
        parts.append(f"雨の可能性{prob:.0f}%。カッパを念のため")
    elif prob >= 20:
        parts.append(f"雨の可能性{prob:.0f}%。にわか雨程度の可能性あり")

    # 風コメント
    if wind >= 12:
        parts.append(f"風速{wind:.1f}m/sの強風。クラブ選択と弾道に要注意")
    elif wind >= 8:
        parts.append(f"風速{wind:.1f}m/sのやや強い風。アゲインスト・フォローを意識して")
    elif wind >= 5:
        parts.append(f"風速{wind:.1f}m/sの風あり。ショートゲームへの影響を考慮")
    elif wind >= 4:
        parts.append(f"風速{wind:.1f}m/sの風を感じる。ロングショットで1番手分の影響あり")
    elif wind >= 3:
        parts.append(f"微風({wind:.1f}m/s)。ほぼ影響なし")
    else:
        parts.append("ほぼ無風で狙いどおりのショットが期待できる")
    if gust is not None and not np.isnan(gust) and gust >= 10 and gust >= wind * 1.6:
        parts.append(f"突風{gust:.0f}m/sに注意（番手選びは強めに）")

    # 雲量コメント
    if prob < 40:
        if cloud <= 20:
            parts.append("快晴で視界良好")
        elif cloud <= 50:
            parts.append("晴れ間が広がり気持ちの良いラウンド")
        elif cloud <= 80:
            parts.append("曇りがちだが直射日光が遮られ逆に快適")
        else:
            parts.append("厚い雲に覆われ薄暗め")

    return "。".join(parts) + "。"


# ---- メイン UI ----
st.title("⛳ ゴルフ場 精密天気予報")
st.caption("世界7機関の気象モデル＋アンサンブル143通り＋気象庁アメダス実測で補正 — 何日先かでモデルの重みを自動で切り替え")

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
        placeholder="例: 成田市、千葉県富里市、35.77, 140.32、GoogleマップのURL",
        label_visibility="collapsed",
        key="keyword_input",
        on_change=_do_search,
    )
with col2:
    search_btn = st.button("🔍 候補を検索", type="secondary", width="stretch")
st.caption("💡 ゴルフ場の正確な位置で予報したいときは、Googleマップでゴルフ場を開いてURLを貼るか、緯度経度を入力してください（山あいのコースは市役所の位置と天気が違うことがあります）")

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
    st.warning("見つかりませんでした。別のキーワードや住所、または緯度経度で試してください。")

# モデル選択
with st.expander(f"⚙️ 使用する気象モデルを選択（デフォルト: 全{len(WEATHER_MODELS)}モデル）"):
    selected_models = {}
    cols = st.columns(4)
    for i, (name, model_id) in enumerate(WEATHER_MODELS.items()):
        with cols[i % 4]:
            if st.checkbox(name, value=True, key=f"model_{model_id}"):
                selected_models[name] = model_id

if not selected_models:
    st.warning("少なくとも1つのモデルを選択してください。")
    st.stop()

if forecast_btn and lat:
    st.success(f"📍 **{course_name}**  |  {address}  |  緯度 {lat:.4f} / 経度 {lon:.4f}")

    try:
        result = load_forecast(round(lat, 4), round(lon, 4), tuple(selected_models.values()))
    except Exception:
        result = None
    if result is None:
        st.error("天気データの取得に失敗しました。しばらく待ってから再試行してください。")
        st.stop()

    ensemble_df = result["df"]
    station, verif = result["station"], result["verif"]

    # --- この地点の検証バナー ---
    if station and verif:
        temp_maes = [v["temp_mae_raw"] for v in verif.values() if "temp_mae_raw" in v]
        wind_maes = [v["wind_mae"] for v in verif.values() if "wind_mae" in v]
        rain_h = max((v.get("rain_hours", 0) for v in verif.values()), default=0)
        st.markdown(f"### 📊 近くのアメダス「{station['name']}」（{station['dist']:.0f}km）で過去7日を検証して補正済み")
        st.markdown(f"""
<div class="acc-banner">
  <div class="acc-card" style="background:#1a3a2f">
    <div class="big">±{min(temp_maes):.1f}〜{max(temp_maes):.1f}°C</div>
    <div class="sub">🌡️ モデル別の気温誤差（補正前）</div>
  </div>
  <div class="acc-card" style="background:#1a3a5c">
    <div class="big">±{min(wind_maes) if wind_maes else float('nan'):.1f}m/s</div>
    <div class="sub">💨 最も当たったモデルの風速誤差</div>
  </div>
  <div class="acc-card" style="background:#2a2a2a">
    <div class="big">{rain_h}時間</div>
    <div class="sub">🌧️ 検証期間に雨が降った時間</div>
  </div>
</div>
""", unsafe_allow_html=True)
        st.caption("※ 数値は直近の予報（数時間〜1日先）での成績です。数日先の予報はこれより外れやすくなります。")
    elif result["jp"]:
        st.info("近く（40km以内）に気温・雨・風を観測するアメダスがないため、実測による補正なしで予報しています。")
    else:
        st.info("日本国外のため、アメダスによる補正と気象庁の降水確率は使わずに予報しています。")

    # モデルの状況と重み
    with st.expander("📡 モデル別の成績と使い方（重み）"):
        rows = []
        for mid in selected_models.values():
            v = verif.get(mid, {})
            ok = mid in result["models_ok"]
            rows.append({
                "モデル": MODEL_NAME.get(mid, mid),
                "取得": "✅" if ok else "❌",
                "気温のクセ": f"{v['temp_bias']:+.1f}°C" if "temp_bias" in v else "-",
                "気温誤差": f"±{v['temp_mae_raw']:.1f}°C" if "temp_mae_raw" in v else "-",
                "風の強さのクセ": f"×{1 / v['wind_ratio']:.2f}" if "wind_ratio" in v else "-",
                "雨の捕捉率": f"{v['rain_pod'] * 100:.0f}%" if "rain_pod" in v else "-",
                "重み(0-3日)": f"{core.prior_weight(mid, 0, result['jp']):.1f}",
                "重み(4-7日)": f"{core.prior_weight(mid, 5, result['jp']):.1f}",
                "重み(8日〜)": f"{core.prior_weight(mid, 9, result['jp']):.1f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "気温のクセ：+は実測より高く出がち（半分を差し引いて補正）。風の強さのクセ：×1.2なら実測より2割強く出がち（半分を補正）。"
            "重み：一般的な予報成績に基づく基本の重み。日本では近い日は気象庁MSM、先の日はECMWFを重視し、さらに上の検証で当たっていたモデルを上げています。"
        )
        st.caption(
            f"雨の確率：決定論モデルの多数決＋アンサンブル{'（取得済み）' if result['ens_ok'] else '（今回は取得失敗）'}"
            + (f"＋気象庁の降水確率（{result['jma_area']}）" if result["jma_area"] else "")
        )

    tabs = st.tabs(["📅 今日", "📅 明日", "📅 明後日", "📆 3〜7日先", "📆 8〜14日先", "📊 全期間グラフ"])

    def render_day_detail(df_day, title, key_suffix=""):
        if df_day.empty:
            st.info("データなし")
            return

        # サマリー指標
        st.markdown(f"### {title}")
        play = df_day[(df_day["time"].dt.hour >= 6) & (df_day["time"].dt.hour <= 17)]
        play = play if not play.empty else df_day
        gust_max = play["windgusts_10m"].max()
        cards_html = "".join([
            f'<div class="summary-card"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
            for l, v in [
                ("最高気温", f"{df_day['temperature_2m'].max():.1f}°C"),
                ("最低気温", f"{df_day['temperature_2m'].min():.1f}°C"),
                ("雨の可能性(6-18時最大)", f"{play['precipitation_probability'].max():.0f}%"),
                ("最大風速(6-18時)", f"{play['windspeed_10m'].max():.1f} m/s"),
                ("最大瞬間風速(6-18時)", f"{gust_max:.0f} m/s" if not np.isnan(gust_max) else "-"),
                ("予想雨量(1日)", f"{df_day['precipitation'].sum():.1f} mm"),
            ]
        ])
        st.markdown(f'<div class="summary-grid">{cards_html}</div>', unsafe_allow_html=True)

        # 気象庁の公式降水確率（今日・明日のみ）
        pops = [(s, e, p) for s, e, p in result["jma_pop"] if s.date() == df_day["time"].dt.date.iloc[0]]
        if pops:
            txt = "　".join(f"{s:%H}-{e:%H}時 **{p:.0f}%**" for s, e, p in pops)
            st.caption(f"🗾 気象庁発表の降水確率（{result['jma_area']}）：{txt}　※この確率に半分寄せて補正済み")

        # 1時間ごとテーブル
        display_df = df_day.copy()
        display_df["時刻"] = display_df["time"].dt.strftime("%H:%M")
        display_df["天気"] = display_df["weathercode"].astype(float).apply(get_weather_code_label)
        display_df["気温"] = display_df["temperature_2m"].round(1).apply(lambda v: f"{v}℃")
        display_df["湿度"] = display_df["relativehumidity_2m"].fillna(0).round(0).astype(int).apply(lambda v: f"{v}%")
        display_df["雨の可能性"] = display_df["precipitation_probability"].fillna(0).astype(int).apply(lambda v: f"{v}%")
        display_df["降れば"] = [
            f"{a:.1f}mm" if p >= 0.2 and a > 0 else "-"
            for p, a in zip(display_df["rain_prob"], display_df["rain_amount"])
        ]
        display_df["風速"] = display_df["windspeed_10m"].round(1).apply(lambda v: f"{v}m/s")
        display_df["瞬間"] = display_df["windgusts_10m"].apply(lambda v: f"{v:.0f}m/s" if not np.isnan(v) else "-")
        display_df["風向"] = display_df["winddirection_10m"].apply(core.wind_dir_label) if "winddirection_10m" in display_df else "-"
        display_df["コメント"] = display_df.apply(generate_hourly_comment, axis=1)

        cols_show = ["時刻", "天気", "気温", "湿度", "雨の可能性", "降れば", "風速", "瞬間", "風向", "コメント"]
        rows_html = ""
        for _, r in display_df[cols_show].iterrows():
            cells = []
            for i, (c, v) in enumerate(r.items()):
                if i == 0:  # 時刻列を横スクロール時に固定
                    cells.append(f'<td style="white-space:nowrap;padding:6px 10px;border-bottom:1px solid #ddd;position:sticky;left:0;background:#ffffff;color:#111111;z-index:1;font-weight:bold;border-right:2px solid #ccc">{v}</td>')
                elif c != "コメント":
                    cells.append(f'<td style="white-space:nowrap;padding:6px 10px;border-bottom:1px solid #ddd">{v}</td>')
                else:
                    cells.append(f'<td style="padding:6px 10px;border-bottom:1px solid #ddd;min-width:200px">{v}</td>')
            rows_html += "<tr>" + "".join(cells) + "</tr>"

        header_cells = []
        for i, c in enumerate(cols_show):
            if i == 0:
                header_cells.append(f'<th style="padding:6px 10px;background:#1e3a2f;color:#ffffff;text-align:left;white-space:nowrap;position:sticky;left:0;z-index:2;border-right:2px solid #4a7a5f">{c}</th>')
            else:
                header_cells.append(f'<th style="padding:6px 10px;background:#1e3a2f;color:#ffffff;text-align:left;white-space:nowrap">{c}</th>')
        header_html = "".join(header_cells)

        st.markdown(f"""
<div style="overflow-x:auto">
<table class="golf-table" style="width:100%;border-collapse:collapse;font-size:13px;color:inherit">
<thead style="position:sticky;top:0;z-index:10"><tr>{header_html}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
""", unsafe_allow_html=True)
        st.caption("雨の可能性＝その1時間に0.2mm以上降る確率。降れば＝降った場合の1時間雨量の目安。")

        # プレー快適スコア（天気スコアのみ）
        st.markdown("#### 🏌️ プレー快適スコア")
        score_df = display_df.copy()
        score_df["スコア"] = score_df.apply(core.play_score, axis=1).astype(int)

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
        st.plotly_chart(gfig, width="stretch", config={"staticPlot": True, "displayModeBar": False}, key=f"gfig_{title}{key_suffix}")
        st.caption("🟢 80点以上: 最高　🟡 60〜79点: 良好　🟠 40〜59点: 注意　🔴 39点以下: 困難　（雨は「確率×降った場合の減点」で計算）")

    today = result["now"].date()

    WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]

    for i in range(3):
        with tabs[i]:
            target_date = today + timedelta(days=i)
            day_df = ensemble_df[ensemble_df["time"].dt.date == target_date]
            label = ["今日", "明日", "明後日"][i]
            dow = WEEKDAYS[target_date.weekday()]
            render_day_detail(day_df, f"{label} ({target_date.strftime('%m/%d')}・{dow})")

    with tabs[3]:
        # 3〜7日先
        start = today + timedelta(days=3)
        end = today + timedelta(days=7)
        future_df = ensemble_df[(ensemble_df["time"].dt.date >= start) & (ensemble_df["time"].dt.date <= end)]
        for date in sorted(future_df["time"].dt.date.unique()):
            day_df = future_df[future_df["time"].dt.date == date]
            dow = WEEKDAYS[date.weekday()]
            render_day_detail(day_df, f"{date.month}月{date.day}日（{dow}）", key_suffix=f"_w1_{date}")
            st.divider()

    with tabs[4]:
        # 8〜14日先
        start = today + timedelta(days=8)
        future_df = ensemble_df[ensemble_df["time"].dt.date >= start]
        if future_df.empty:
            st.info("8日以降のデータがまだ取得できていません。")
        else:
            st.caption("⚠️ 8日以上先は予報の当たりやすさが大きく下がります。傾向（雨が多そうか・寒そうか）の目安にしてください。")
            for date in sorted(future_df["time"].dt.date.unique()):
                day_df = future_df[future_df["time"].dt.date == date]
                dow = WEEKDAYS[date.weekday()]
                render_day_detail(day_df, f"{date.month}月{date.day}日（{dow}）", key_suffix=f"_w2_{date}")
                st.divider()

    # --- 全期間グラフ ---
    with tabs[5]:
        st.markdown("### 14日間 1時間ごと推移グラフ")

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "🌡️ 気温 (°C)",
                "🌧️ 雨の可能性 (%) / 予想雨量 (mm)",
                "💨 風速・瞬間風速 (m/s)",
            ),
            row_heights=[0.35, 0.35, 0.30],
            specs=[[{}], [{"secondary_y": True}], [{}]],
        )

        x = ensemble_df["time"]

        # 気温
        fig.add_trace(go.Scatter(x=x, y=ensemble_df["temperature_2m"],
            name="気温", line=dict(color="#ff6b35", width=2.5)), row=1, col=1)
        # 不確実性バンド
        if "temp_std" in ensemble_df:
            std = ensemble_df["temp_std"].fillna(0)
            fig.add_trace(go.Scatter(
                x=pd.concat([x, x[::-1]]),
                y=pd.concat([
                    ensemble_df["temperature_2m"] + std,
                    (ensemble_df["temperature_2m"] - std)[::-1]
                ]),
                fill="toself", fillcolor="rgba(255,107,53,0.15)",
                line=dict(color="rgba(255,107,53,0)"), name="モデル間のばらつき", showlegend=True
            ), row=1, col=1)

        # 雨
        fig.add_trace(go.Bar(x=x, y=ensemble_df["precipitation"],
            name="予想雨量(mm)", marker_color="rgba(0,119,182,0.6)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=ensemble_df["precipitation_probability"],
            name="雨の可能性(%)", line=dict(color="#4cc9f0", width=2)), row=2, col=1, secondary_y=True)
        fig.update_yaxes(range=[0, 100], row=2, col=1, secondary_y=True)

        # 風速
        fig.add_trace(go.Scatter(x=x, y=ensemble_df["windspeed_10m"],
            name="風速(m/s)", line=dict(color="#06d6a0", width=2),
            fill="tozeroy", fillcolor="rgba(6,214,160,0.15)"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=ensemble_df["windgusts_10m"],
            name="瞬間風速(m/s)", line=dict(color="#06d6a0", width=1, dash="dot")), row=3, col=1)
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

        st.plotly_chart(fig, width="stretch", config={"staticPlot": True, "displayModeBar": False})

    st.caption(
        f"🔄 データ更新: {result['now']:%Y-%m-%d %H:%M}（現地時刻） | 使用モデル数: {len(result['models_ok'])}/{len(selected_models)}"
        + (f" | 予報地点の標高 {result['elevation']:.0f}m" if result["elevation"] is not None else "")
    )

else:
    st.markdown(f"""
    ---
    ### 使い方
    1. 上の検索欄に **住所・市区町村名**、または **緯度経度 / GoogleマップのURL** を入力する（例: **成田市**、**千葉県富里市**、**35.77, 140.32**）
    2. 「候補を検索」ボタンを押すと候補がプルダウンで表示される
    3. リストから行きたい場所を選んで「この場所の予報を取得」をクリック

    ### 精度を上げている仕組み
    - **何日先かでモデルを使い分け**：日本では0〜3日先は気象庁MSM（5km格子）、4日目以降は世界最高水準のECMWFを重視
    - **アメダス実測でクセを補正**：最寄りのアメダスの過去7日と各モデルを比べ、気温・風のクセを補正し、当たっていたモデルの重みを上げる
    - **雨は平均しない**：「降る確率」と「降った場合の量」に分けて出す。確率は7モデルの多数決＋アンサンブル143通り＋気象庁の降水確率（今日・明日）で決める
    - 過去データでの検証（全国12地点・プレー時間帯）：旧方式に比べ、雨の確率の誤差が約半分、気温誤差 0.89→0.82°C、風速誤差 1.02→0.92m/s

    **使用している気象モデル：**
    | モデル | 提供機関 | 特徴 |
    |---|---|---|
    | JMA MSM/GSM | 日本気象庁 | 日本域5km格子・3日先まで最有力 |
    | ECMWF IFS 9km | 欧州中期予報センター | 世界最高精度の中期予報 |
    | UKMO | 英国気象局 | 全球10km・ECMWFに次ぐ精度 |
    | ICON | 独DWD | 全球13km |
    | GFS | 米国NOAA | 16日先まで |
    | GEM | カナダ環境省 | 全球 |
    | ARPEGE | フランス気象局 | 全球 |
    | アンサンブル | ECMWF/NOAA/DWD/カナダ | 計143通りの予報で雨の確率を算出 |
    """)
