"""予報の計算ロジック（Streamlitに依存しない部分）

精度を上げる仕組み:
1. 予報モデルごとに「何日先か」で重みを変える（近い日=気象庁MSM、先の日=ECMWF）
2. 日本国内では最寄りアメダスの実測（過去7日）と各モデルを比べ、
   ・気温のクセ（いつも高め/低め）を差し引く
   ・風速のクセ（強め/弱め）を半分だけ補正する
   ・気温/風/雨それぞれ、よく当たっていたモデルの重みを上げる
3. 雨は平均しない。「降る確率」と「降った場合の量」に分けて出す
   ・確率 = 決定論モデルの重み付き多数決 + アンサンブル143通りの割合
   ・日本では今日・明日について気象庁(予報官)の降水確率に半分寄せる
"""
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

JST = timezone(timedelta(hours=9))

# 決定論モデル（Open-Meteo の model id）
WEATHER_MODELS = {
    "JMA (気象庁 MSM/GSM)": "jma_seamless",
    "ECMWF IFS 9km (欧州中期予報センター)": "ecmwf_ifs",
    "UKMO (英国気象局)": "ukmo_seamless",
    "ICON (独DWD)": "icon_seamless",
    "GFS (米NOAA)": "gfs_seamless",
    "GEM (カナダ)": "gem_seamless",
    "ARPEGE (仏Météo-France)": "meteofrance_seamless",
}

# 一般に知られている予報成績を元にした事前の重み（何日先か別）
# 日本国内: 0-3日は気象庁MSM(5km)が最有力、4日目以降はECMWFが世界最高
PRIOR_WEIGHTS_JP = {
    #                0-3日  4-7日  8日〜
    "jma_seamless":        (3.0, 1.5, 1.0),
    "ecmwf_ifs":           (2.0, 3.0, 3.0),
    "ukmo_seamless":       (1.5, 1.8, 1.5),
    "icon_seamless":       (1.0, 1.2, 1.0),
    "gfs_seamless":        (0.8, 1.0, 1.0),
    "gem_seamless":        (0.8, 0.9, 1.0),
    "meteofrance_seamless": (1.0, 1.0, 1.0),
}
PRIOR_WEIGHTS_WORLD = {
    "jma_seamless":        (0.8, 0.8, 0.8),
    "ecmwf_ifs":           (2.5, 3.0, 3.0),
    "ukmo_seamless":       (1.5, 1.8, 1.5),
    "icon_seamless":       (1.2, 1.2, 1.0),
    "gfs_seamless":        (1.0, 1.0, 1.0),
    "gem_seamless":        (0.9, 0.9, 1.0),
    "meteofrance_seamless": (1.0, 1.0, 1.0),
}

# アンサンブル（メンバー数: ECMWF 51 / GEFS 31 / ICON 40 / GEM 21）
ENSEMBLE_MODELS = "ecmwf_ifs025,gfs025,icon_seamless,gem_global"
ENSEMBLE_GROUP_WEIGHT = {"ecmwf": 2.0}  # その他は1.0

RAIN_THRESHOLD = 0.2  # mm/h 以上を「雨」とみなす
TEMP_BIAS_SHRINK = 0.5  # 7日分のクセは揺れがあるので半分だけ使う（過去データ検証で最良）
MIN_RAIN_HOURS_FOR_SKILL = 15  # 雨の重み付けは、検証期間に雨が十分降った時だけ

HOURLY_VARS = [
    "temperature_2m", "apparent_temperature", "precipitation", "weathercode",
    "windspeed_10m", "winddirection_10m", "windgusts_10m", "cloudcover",
    "relativehumidity_2m",
]


def _get(url, params=None, timeout=30, tries=3):
    """一時的なタイムアウト・混雑(429/5xx)は少し待って再試行"""
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code}", response=r)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last = e
            if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code < 429:
                raise
            time.sleep(1.5 * (i + 1))
    raise last


def is_japan(lat, lon):
    return 24.0 <= lat <= 46.0 and 122.5 <= lon <= 146.5


def _lead_bucket(lead_days):
    return 0 if lead_days < 3.5 else (1 if lead_days < 7.5 else 2)


def prior_weight(model_id, lead_days, jp):
    table = PRIOR_WEIGHTS_JP if jp else PRIOR_WEIGHTS_WORLD
    return table.get(model_id, (1.0, 1.0, 1.0))[_lead_bucket(lead_days)]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------
# 取得
# ---------------------------------------------------------------
def _split_models(hourly, model_ids, variables):
    """Open-Meteo の複数モデル応答（temperature_2m_jma_seamless 形式）を縦持ちDFにする"""
    times = pd.to_datetime(hourly["time"])
    frames = []
    for mid in model_ids:
        cols = {}
        for v in variables:
            key = f"{v}_{mid}" if len(model_ids) > 1 else v
            if key in hourly:
                cols[v] = pd.to_numeric(pd.Series(hourly[key]), errors="coerce").values
        if not cols:
            continue
        df = pd.DataFrame(cols)
        if df.drop(columns=[c for c in ["weathercode"] if c in df]).isna().all().all():
            continue
        df.insert(0, "time", times)
        df["model"] = mid
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_forecasts(lat, lon, model_ids):
    """全モデルを1リクエストで取得。戻り値: (縦持ちDF, utcオフセット秒, 標高)"""
    r = _get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(HOURLY_VARS),
            "models": ",".join(model_ids),
            "forecast_days": 14, "timezone": "auto", "wind_speed_unit": "ms",
        },
        timeout=30,
    )
    d = r.json()
    return _split_models(d["hourly"], list(model_ids), HOURLY_VARS), d.get("utc_offset_seconds", 0), d.get("elevation")


def fetch_ensemble_precip(lat, lon):
    """アンサンブル各メンバーの雨量 → 時刻×メンバーのDFと、メンバー→グループ対応"""
    r = _get(
        "https://ensemble-api.open-meteo.com/v1/ensemble",
        params={
            "latitude": lat, "longitude": lon, "hourly": "precipitation",
            "models": ENSEMBLE_MODELS, "forecast_days": 14, "timezone": "auto",
        },
        timeout=30,
    )
    h = r.json()["hourly"]
    times = pd.to_datetime(h.pop("time"))
    data, groups = {}, {}
    for k, v in h.items():
        data[k] = pd.to_numeric(pd.Series(v), errors="coerce").values
        groups[k] = re.sub(r"member\d+_", "", k).replace("precipitation_", "")
    return pd.DataFrame(data, index=times), groups


def fetch_jma_official_pop(lat, lon):
    """気象庁の府県予報（予報官が出す6時間ごとの降水確率、今日〜明日）。
    最寄りの代表アメダス地点から一次細分区域を決める。戻り値: [(開始時刻, 終了時刻, 確率%)], 区域名"""
    areas = _get("https://www.jma.go.jp/bosai/forecast/const/forecast_area.json", timeout=15).json()
    table = fetch_amedas_table()
    best = None
    for office, lst in areas.items():
        for item in lst:
            for code in item.get("amedas", []):
                st_ = table.get(code)
                if not st_:
                    continue
                d = haversine_km(lat, lon, st_["lat"], st_["lon"])
                if best is None or d < best[0]:
                    best = (d, office, item["class10"])
    if best is None or best[0] > 80:
        return [], None
    _, office, class10 = best
    fc = _get(f"https://www.jma.go.jp/bosai/forecast/data/forecast/{office}.json", timeout=15).json()
    ts = fc[0]["timeSeries"][1]
    area = next((a for a in ts["areas"] if a["area"]["code"] == class10), None)
    if area is None:
        return [], None
    out = []
    for t, p in zip(ts["timeDefines"], area["pops"]):
        if p == "":
            continue
        start = pd.Timestamp(t).tz_convert("Asia/Tokyo").tz_localize(None)
        out.append((start, start + pd.Timedelta(hours=6), float(p)))
    return out, area["area"]["name"]


# ---- アメダス ----
_AMEDAS_TABLE = None


def fetch_amedas_table():
    global _AMEDAS_TABLE
    if _AMEDAS_TABLE is None:
        raw = _get("https://www.jma.go.jp/bosai/amedas/const/amedastable.json", timeout=15).json()
        _AMEDAS_TABLE = {
            code: {
                "name": v["kjName"],
                "lat": v["lat"][0] + v["lat"][1] / 60,
                "lon": v["lon"][0] + v["lon"][1] / 60,
                "alt": v.get("alt", 0),
                "elems": v.get("elems", ""),
            }
            for code, v in raw.items()
        }
    return _AMEDAS_TABLE


def find_amedas_station(lat, lon, max_km=40):
    """気温・雨・風をすべて観測している最寄りのアメダス"""
    best = None
    for code, s in fetch_amedas_table().items():
        e = s["elems"]
        if len(e) < 3 or e[:3] != "111":
            continue
        d = haversine_km(lat, lon, s["lat"], s["lon"])
        if best is None or d < best["dist"]:
            best = dict(s, code=code, dist=d)
    return best if best and best["dist"] <= max_km else None


def fetch_amedas_obs(code, start_date, end_date):
    """アメダスの毎正時の実測（気温・前1時間雨量・風速）。日時はJST"""
    urls = []
    d = start_date
    while d <= end_date:
        for hh in range(0, 24, 3):
            urls.append(f"https://www.jma.go.jp/bosai/amedas/data/point/{code}/{d:%Y%m%d}_{hh:02d}.json")
        d += timedelta(days=1)

    def get(u):
        try:
            r = requests.get(u, timeout=15)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for blob in ex.map(get, urls):
            for key, rec in blob.items():
                if key[10:12] != "00":  # 正時だけ使う
                    continue

                def val(name):
                    v = rec.get(name)
                    return v[0] if v and v[0] is not None and v[1] == 0 else np.nan

                rows.append({
                    "time": pd.Timestamp(datetime.strptime(key, "%Y%m%d%H%M%S")),
                    "temperature_2m": val("temp"),
                    "precipitation": val("precipitation1h"),
                    "windspeed_10m": val("wind"),
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fetch_past_forecasts(lat, lon, elevation, model_ids, start_date, end_date):
    """過去の予報（各時刻の直近の予報）。アメダス地点の標高に合わせて取得"""
    r = _get(
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon, "elevation": elevation,
            "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
            "hourly": "temperature_2m,precipitation,windspeed_10m",
            "models": ",".join(model_ids),
            "timezone": "Asia/Tokyo", "wind_speed_unit": "ms",
        },
        timeout=30,
    )
    return _split_models(r.json()["hourly"], list(model_ids), ["temperature_2m", "precipitation", "windspeed_10m"])


# ---------------------------------------------------------------
# 検証（アメダス vs 各モデル）
# ---------------------------------------------------------------
def verify_models(obs, past):
    """モデルごとのクセと誤差。戻り値: {model_id: {...}}"""
    out = {}
    if obs.empty or past.empty:
        return out
    for mid, g in past.groupby("model"):
        m = obs.merge(g, on="time", suffixes=("_obs", "_fc"))
        res = {"hours": len(m)}
        t = m[["temperature_2m_obs", "temperature_2m_fc"]].dropna()
        if len(t) >= 24:
            err = t["temperature_2m_fc"] - t["temperature_2m_obs"]
            res["temp_bias"] = float(err.mean())
            res["temp_mae_raw"] = float(err.abs().mean())
            res["temp_mae"] = float((err - err.mean()).abs().mean())
        w = m[["windspeed_10m_obs", "windspeed_10m_fc"]].dropna()
        if len(w) >= 24 and w["windspeed_10m_fc"].mean() > 0.3:
            ratio = w["windspeed_10m_obs"].mean() / w["windspeed_10m_fc"].mean()
            res["wind_ratio"] = float(np.clip(ratio, 0.5, 2.0))
            adj = w["windspeed_10m_fc"] * (1 + 0.5 * (res["wind_ratio"] - 1))
            res["wind_mae"] = float((adj - w["windspeed_10m_obs"]).abs().mean())
        p = m[["precipitation_obs", "precipitation_fc"]].dropna()
        if len(p) >= 24:
            o = p["precipitation_obs"] >= 0.5
            f = p["precipitation_fc"] >= RAIN_THRESHOLD
            res["rain_hours"] = int(o.sum())
            res["precip_mae"] = float((p["precipitation_fc"] - p["precipitation_obs"]).abs().mean())
            if o.sum() >= 3:
                res["rain_pod"] = float((o & f).sum() / o.sum())       # 降った時間を当てた率
            if f.sum() >= 3:
                res["rain_far"] = float((~o & f).sum() / f.sum())      # 雨予報の空振り率
        out[mid] = res
    return out


def skill_factors(verif, key):
    """誤差が小さいモデルほど大きい係数（0.5〜2.0）。データ不足は1.0"""
    maes = {m: v[key] for m, v in verif.items() if key in v and v[key] > 0}
    if len(maes) < 2:
        return {}
    med = float(np.median(list(maes.values())))
    return {m: float(np.clip(med / e, 0.5, 2.0)) for m, e in maes.items()}


def precip_skill(verif):
    """雨は「強さの誤差」と「降った時間の捕捉/空振り」を合わせて評価"""
    if not verif or max(v.get("rain_hours", 0) for v in verif.values()) < MIN_RAIN_HOURS_FOR_SKILL:
        return {}
    base = skill_factors(verif, "precip_mae")
    out = {}
    for m, v in verif.items():
        f = base.get(m, 1.0)
        if "rain_pod" in v:
            f *= 0.6 + 0.8 * v["rain_pod"]          # 捕捉率 0→0.6倍, 1→1.4倍
        if "rain_far" in v:
            f *= 1.3 - 0.6 * v["rain_far"]          # 空振り率 0→1.3倍, 1→0.7倍
        out[m] = float(np.clip(f, 0.4, 2.5))
    return out


# ---------------------------------------------------------------
# 合成
# ---------------------------------------------------------------
def _weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2)])


def _derive_weathercode(prob, amount, cloud, temp, codes, cw):
    """合成した雨確率・雲量から天気を決める（モデルの天気コードの多数決はしない）"""
    codes = np.asarray(codes)
    cw = np.asarray(cw)
    tot = cw.sum() or 1
    thunder = cw[codes >= 95].sum() / tot
    fog = cw[(codes == 45) | (codes == 48)].sum() / tot
    if prob >= 0.5:
        if thunder >= 0.3:
            return 95
        if not np.isnan(temp) and temp <= 1.0:
            return 73 if amount >= 1 else 71
        return 65 if amount >= 4 else (63 if amount >= 1 else 61)
    if prob >= 0.3:
        return 80
    if fog >= 0.5:
        return 45
    if np.isnan(cloud):
        return 2
    return 0 if cloud < 20 else (1 if cloud < 50 else (2 if cloud < 80 else 3))


def build_blend(fc, now_local, jp, verif=None, ens=None, ens_groups=None, jma_pop=None):
    """全モデルを合成した1時間ごとの予報DFを返す"""
    verif = verif or {}
    sk_t = skill_factors(verif, "temp_mae")
    sk_w = skill_factors(verif, "wind_mae")
    sk_p = precip_skill(verif)

    fc = fc.copy()
    fc["lead"] = ((fc["time"] - now_local).dt.total_seconds() / 86400).clip(lower=0)

    # 検証の強さ: 検証は「直近の予報」の成績なので、先の日ほど効きを弱める
    alpha = np.where(fc["lead"] < 2.5, 1.0, np.where(fc["lead"] < 6.5, 0.5, 0.25))
    decay = np.clip(1 - (fc["lead"] - 3) / 22, 0.5, 1.0)  # クセ補正: 3日先まで全量→14日先で半分

    prior = np.array([prior_weight(m, l, jp) for m, l in zip(fc["model"], fc["lead"])])
    fc["w_t"] = prior * np.array([sk_t.get(m, 1.0) for m in fc["model"]]) ** alpha
    fc["w_w"] = prior * np.array([sk_w.get(m, 1.0) for m in fc["model"]]) ** alpha
    fc["w_p"] = prior * np.array([sk_p.get(m, 1.0) for m in fc["model"]]) ** alpha

    # クセ補正（気温は検証で半分だけ補正が最も当たったので0.5倍）
    bias = fc["model"].map(lambda m: verif.get(m, {}).get("temp_bias", 0.0)).astype(float) * TEMP_BIAS_SHRINK
    fc["temperature_2m"] = fc["temperature_2m"] - bias * decay
    if "apparent_temperature" in fc:
        fc["apparent_temperature"] = fc["apparent_temperature"] - bias * decay
    ratio = fc["model"].map(lambda m: verif.get(m, {}).get("wind_ratio", 1.0)).astype(float)
    wind_k = 1 + 0.5 * (ratio - 1) * decay
    fc["windspeed_10m"] = fc["windspeed_10m"] * wind_k
    if "windgusts_10m" in fc:
        fc["windgusts_10m"] = fc["windgusts_10m"] * wind_k

    # アンサンブル: 時刻ごとの雨の割合と、降った場合の量
    ens_prob, ens_amt = {}, {}
    if ens is not None and not ens.empty:
        cols = list(ens.columns)
        gw = np.array([next((v for k, v in ENSEMBLE_GROUP_WEIGHT.items() if k in ens_groups[c]), 1.0) for c in cols])
        # グループごとにメンバー数で割って「1グループ=重み」にする
        counts = pd.Series([ens_groups[c] for c in cols]).value_counts()
        mw = gw / np.array([counts[ens_groups[c]] for c in cols])
        arr = ens.values
        for i, t in enumerate(ens.index):
            row = arr[i]
            ok = ~np.isnan(row)
            if ok.sum() < 10:
                continue
            wet = row[ok] >= RAIN_THRESHOLD
            ens_prob[t] = float((mw[ok] * wet).sum() / mw[ok].sum())
            if wet.any():
                ens_amt[t] = _weighted_median(row[ok][wet], mw[ok][wet])

    rows = []
    for t, g in fc.groupby("time"):
        lead = float(g["lead"].iloc[0])
        row = {"time": t, "lead": lead}

        def wmean(col, wcol):
            if col not in g:
                return np.nan
            v = g[col].values.astype(float)
            w = g[wcol].values
            ok = ~np.isnan(v)
            return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan

        row["temperature_2m"] = wmean("temperature_2m", "w_t")
        row["apparent_temperature"] = wmean("apparent_temperature", "w_t")
        row["relativehumidity_2m"] = wmean("relativehumidity_2m", "w_t")
        row["cloudcover"] = wmean("cloudcover", "w_t")
        row["windspeed_10m"] = wmean("windspeed_10m", "w_w")
        row["windgusts_10m"] = wmean("windgusts_10m", "w_w")

        # 気温のばらつき（重み付き標準偏差）
        tv, tw = g["temperature_2m"].values.astype(float), g["w_t"].values
        ok = ~np.isnan(tv)
        if ok.sum() >= 2:
            mu = np.average(tv[ok], weights=tw[ok])
            row["temp_std"] = float(np.sqrt(np.average((tv[ok] - mu) ** 2, weights=tw[ok])))
        else:
            row["temp_std"] = np.nan

        # 風向は東西・南北成分に分けて平均（350°と10°の平均が180°にならないように）
        if "winddirection_10m" in g:
            d = np.radians(g["winddirection_10m"].values.astype(float))
            s = g["windspeed_10m"].values.astype(float)
            w = g["w_w"].values
            ok = ~np.isnan(d) & ~np.isnan(s)
            if ok.any():
                u = np.average(s[ok] * np.sin(d[ok]), weights=w[ok])
                v = np.average(s[ok] * np.cos(d[ok]), weights=w[ok])
                row["winddirection_10m"] = float(np.degrees(np.arctan2(u, v)) % 360)

        # 雨: 決定論モデルの重み付き多数決
        pv, pw = g["precipitation"].values.astype(float), g["w_p"].values
        ok = ~np.isnan(pv)
        det_prob, det_amt = np.nan, np.nan
        if ok.any():
            wet = pv[ok] >= RAIN_THRESHOLD
            det_prob = float(pw[ok][wet].sum() / pw[ok].sum())
            if wet.any():
                det_amt = _weighted_median(pv[ok][wet], pw[ok][wet])
            row["models_wet"] = int(wet.sum())
            row["models_n"] = int(ok.sum())

        e_prob = ens_prob.get(t, np.nan)
        # 近い日は高解像度の決定論モデル、先の日はアンサンブルを重視
        k_det = 0.65 if lead < 2.5 else (0.4 if lead < 6.5 else 0.25)
        if np.isnan(det_prob):
            prob = e_prob
        elif np.isnan(e_prob):
            prob = det_prob
        else:
            prob = k_det * det_prob + (1 - k_det) * e_prob
        row["ens_prob"] = e_prob
        row["rain_prob"] = 0.0 if np.isnan(prob) else prob
        amt = det_amt if not np.isnan(det_amt) else ens_amt.get(t, np.nan)
        row["rain_amount"] = 0.0 if np.isnan(amt) else amt  # 降った場合の1時間雨量

        row["_codes"] = g["weathercode"].fillna(0).values if "weathercode" in g else np.array([0])
        row["_cw"] = g["w_p"].values if "weathercode" in g else np.array([1.0])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    # 気象庁（予報官）の6時間降水確率に半分寄せる（今日・明日）
    df["jma_pop"] = np.nan
    if jma_pop:
        for start, end, pop in jma_pop:
            mask = (df["time"] > start) & (df["time"] <= end)
            if not mask.any():
                continue
            df.loc[mask, "jma_pop"] = pop
            block_max = df.loc[mask, "rain_prob"].max()
            shift = 0.5 * (pop / 100 - block_max)
            df.loc[mask, "rain_prob"] = (df.loc[mask, "rain_prob"] + shift).clip(0, 1)

    df["precipitation"] = df["rain_prob"] * df["rain_amount"]  # 期待雨量
    df["precipitation_probability"] = (df["rain_prob"] * 100).round()
    df["weathercode"] = [
        _derive_weathercode(p, a, c, tt, codes, cw)
        for p, a, c, tt, codes, cw in zip(df["rain_prob"], df["rain_amount"], df["cloudcover"],
                                          df["temperature_2m"], df["_codes"], df["_cw"])
    ]
    return df.drop(columns=["_codes", "_cw"])


def rain_penalty(amount):
    if amount >= 4:
        return 80
    if amount >= 2:
        return 60
    if amount >= 1:
        return 40
    if amount >= 0.3:
        return 30
    if amount > 0:
        return 10
    return 0


def play_score(row, include_rain=True):
    """プレー快適スコア。雨は「確率 × 降った場合の減点」で期待値にする"""
    s = 100
    prob = (row.get("rain_prob", 0) or 0) if include_rain else 0
    amt = row.get("rain_amount", 0) or 0
    ws = row.get("windspeed_10m", 0) or 0
    gust = row.get("windgusts_10m", np.nan)
    t = row.get("temperature_2m", 20)
    t = 20 if t is None or np.isnan(t) else t
    rh = row.get("relativehumidity_2m", 50)
    rh = 50 if rh is None or np.isnan(rh) else rh
    s -= prob * rain_penalty(amt)
    if ws >= 8:
        s -= 100
    elif ws >= 6:
        s -= 50
    elif ws >= 4:
        s -= 30
    if gust is not None and not np.isnan(gust):
        if gust >= 15:
            s -= 20
        elif gust >= 12:
            s -= 10
    if 15 <= t <= 25:
        s += 5
    elif t < 5 or t > 35:
        s -= 25
    elif t < 10 or t > 30:
        s -= 12
    if t >= 25:
        if rh >= 85:
            s -= 20
        elif rh >= 75:
            s -= 10
    return int(round(max(0, min(100, s))))


def day_score(play):
    """1日（プレー時間帯）の点数。
    1時間ごとの点数を平均すると、数時間の雨が薄まって雨の日でも高得点になるので、
    雨以外（風・気温・湿度）の平均から「雨に当たりそうな時間」に応じて差し引く。
    例: 1mm/h級の雨が確実に1時間 → -10点、ほぼ1日中雨 → -80点（上限）"""
    if play.empty:
        return 0
    base = play.apply(lambda r: play_score(r, include_rain=False), axis=1).mean()
    wet_hours = sum(
        p * rain_penalty(a) / 40
        for p, a in zip(play["rain_prob"].fillna(0), play["rain_amount"].fillna(0))
    )
    return int(round(max(0, min(100, base - min(80, 10 * wet_hours)))))


WIND_DIRS = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
             "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]


def wind_dir_label(deg):
    if deg is None or np.isnan(deg):
        return "-"
    return WIND_DIRS[int((deg + 11.25) // 22.5) % 16]


def parse_coords(text):
    """「35.77, 140.32」や Google マップのURL（@35.77,140.32 / !3d35.77!4d140.32）から緯度経度を取り出す"""
    m = (re.search(r"!3d(-?\d{1,2}\.\d+)!4d(-?\d{1,3}\.\d+)", text)
         or re.search(r"@(-?\d{1,2}\.\d+),(-?\d{1,3}\.\d+)", text)
         or re.search(r"[?&](?:q|query|ll)=(-?\d{1,2}\.\d+),\s*(-?\d{1,3}\.\d+)", text))
    if not m and "http" not in text and "google" not in text:
        m = re.search(r"(-?\d{1,2}\.\d{2,})\s*[,、 ]\s*(-?\d{1,3}\.\d{2,})", text)
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None


def expand_short_url(text):
    """maps.app.goo.gl / goo.gl の短縮URLを展開する（失敗したらそのまま返す）"""
    m = re.search(r"https?://(?:maps\.app\.goo\.gl|goo\.gl)/\S+", text)
    if not m:
        return text
    try:
        r = requests.get(m.group(0), allow_redirects=True, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        return r.url
    except Exception:
        return text


def place_name_from_maps_url(text):
    """GoogleマップのURL（/maps/place/マリアゴルフ/data=...）から場所の名前を取り出す"""
    from urllib.parse import unquote_plus
    m = re.search(r"/maps/place/([^/?@]+)", text)
    if m:
        return unquote_plus(m.group(1)).strip()
    m = re.search(r"[?&]q=([^&]+)", text)
    if m and "google" in text:
        return unquote_plus(m.group(1)).strip()
    return None


_MUNI = None


def _muni_table():
    """国土地理院の市区町村コード表（12206 → 千葉県 木更津市）"""
    global _MUNI
    if _MUNI is None:
        _MUNI = {}
        try:
            r = _get("https://maps.gsi.go.jp/js/muni.js", timeout=15)
            r.encoding = "utf-8"
            for code, val in re.findall(r"MUNI_ARRAY\[\"(\d+)\"\]\s*=\s*'([^']+)'", r.text):
                parts = val.split(",")
                if len(parts) >= 4:
                    _MUNI[code.lstrip("0")] = f"{parts[1]} {parts[3]}"
        except Exception:
            pass
    return _MUNI


def gsi_search(keyword, limit=10):
    """国土地理院の地名検索（住所に加え、ゴルフ場などの施設名も引ける）"""
    r = _get("https://msearch.gsi.go.jp/address-search/AddressSearch", params={"q": keyword}, timeout=10)
    items = r.json() or []
    muni = _muni_table()
    kw = keyword.replace(" ", "").replace("　", "")
    out, seen = [], set()
    for it in items:
        lon, lat = it["geometry"]["coordinates"][:2]
        title = it["properties"].get("title", "").strip()
        key = (round(lat, 3), round(lon, 3))
        if key in seen or not title:
            continue
        seen.add(key)
        code = (it["properties"].get("addressCode") or "").lstrip("0")
        out.append({
            "name": title, "lat": float(lat), "lon": float(lon),
            "address": muni.get(code, "") or "国土地理院",
            "_exact": kw in title.replace(" ", ""),
        })
    # 入力した名前をそのまま含むもの（施設名の一致）を先に
    out.sort(key=lambda x: not x["_exact"])
    for x in out:
        x.pop("_exact")
    return out[:limit]


def now_local(utc_offset_seconds):
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) + pd.Timedelta(seconds=utc_offset_seconds)
