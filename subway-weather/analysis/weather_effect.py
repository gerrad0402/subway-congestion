"""날씨가 지하철 이용량에 미치는 영향 추정.

방법
 1. 시간별 승차 인원(전 호선 합)에서 '같은 달·같은 요일그룹·같은 시간'의 중앙값을 기대치로 잡고,
    ratio = 실제 / 기대치 를 만든다. (계절·요일·시간대 패턴 제거)
 2. 시간별 날씨(비, 눈, 한파, 폭염)를 같은 시간에 붙여 log(ratio)를 회귀한다.
 3. 시간대 묶음(출근/퇴근/낮/저녁)별로 '날씨 조건이 있으면 이용량이 몇 % 변하는가'를 JSON으로 저장한다.

주의: 상관관계이며 인과가 아닙니다. 기대치가 같은 달 중앙값이라 '계절 평균 기온' 효과는 빠지고
      '그 달 평소보다 춥다/덥다' 같은 편차와 강수·적설 효과만 남습니다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pipeline.ridership import flag_offdays

RIDERSHIP = Path("data/processed/ridership_hourly.csv")
WEATHER = Path("data/raw/asos_seoul_clean.csv")
OUT = Path("models/weather_effect.json")

BANDS = {"출근(7-9시)": range(7, 10), "퇴근(17-20시)": range(17, 20),
         "낮(10-16시)": range(10, 17), "저녁(20-23시)": range(20, 24)}
THRESH = {"rain_heavy_mm": 5.0, "cold_c": -5.0, "hot_c": 30.0}
FEATURES = ["rain", "rain_heavy", "snow", "cold", "hot"]


def day_group(ts: pd.Series, offday: pd.Series) -> pd.Series:
    wd = ts.dt.weekday
    g = pd.Series("tue_thu", index=ts.index)
    g[wd == 0] = "mon"
    g[wd == 4] = "fri"
    g[wd == 5] = "sat"
    g[wd == 6] = "sun"
    g[offday] = "holiday"
    return g


def build_ratio(rid: pd.DataFrame) -> pd.DataFrame:
    """승차 인원(전 호선 합) 시간별 기대치 대비 비율."""
    b = rid[rid["io"] == "승차"].groupby(["date", "hour"], as_index=False)["count"].sum()
    daily = b.groupby("date")["count"].sum()
    off = flag_offdays(daily).rename("offday")
    b = b.join(off, on="date")
    b["group"] = day_group(b["date"], b["offday"])
    b["ym"] = b["date"].dt.to_period("M")
    exp = b.groupby(["ym", "group", "hour"])["count"].transform("median")
    b["ratio"] = b["count"] / exp
    return b


def attach_weather(b: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """승차 시간대 H(H시~H+1시)에는 관측시각 H+1시의 값을 붙인다 (강수량은 직전 1시간 누적)."""
    w = w.copy()
    w["obs_time"] = pd.to_datetime(w["obs_time"])
    w["slot"] = w["obs_time"] - pd.Timedelta(hours=1)
    w["date"] = w["slot"].dt.normalize()
    w["hour"] = w["slot"].dt.hour
    return b.merge(w.drop(columns=["obs_time", "slot"]), on=["date", "hour"], how="left")


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rn = df["rain_mm"].fillna(0)
    df["rain"] = ((rn > 0) & (rn < THRESH["rain_heavy_mm"])).astype(int)
    df["rain_heavy"] = (rn >= THRESH["rain_heavy_mm"]).astype(int)
    sn = df.get("snow_3h", pd.Series(0, index=df.index)).fillna(0)
    sd = df.get("snow_depth", pd.Series(0, index=df.index)).fillna(0)
    df["snow"] = ((sn > 0) | (sd > 0)).astype(int)
    df["cold"] = (df["temp"] <= THRESH["cold_c"]).astype(int)
    df["hot"] = (df["temp"] >= THRESH["hot_c"]).astype(int)
    return df


def fit(df: pd.DataFrame) -> dict:
    """시간대 묶음 × (평일/주말) 별 효과. 휴일과 이상치 비율은 제외."""
    out = {}
    d = df[(df["group"] != "holiday") & df["ratio"].between(0.3, 2.0)].dropna(subset=["temp"])
    d = d.assign(daytype=np.where(d["group"].isin(["sat", "sun"]), "weekend", "weekday"))
    for dt_name, dsub in d.groupby("daytype"):
        for name, hours in BANDS.items():
            s = dsub[dsub["hour"].isin(hours)]
            if len(s) < 200:
                continue
            X, y = s[FEATURES], np.log(s["ratio"])
            m = Ridge(alpha=1.0).fit(X, y)
            r2 = m.score(X, y)
            out.setdefault(dt_name, {})[name] = {
                "n": int(len(s)),
                "r2": round(float(r2), 4),
                # log 계수 → 이용량 변화율(%)
                "effect_pct": {f: round(float((np.exp(c) - 1) * 100), 2) for f, c in zip(FEATURES, m.coef_)},
                "n_flag": {f: int(s[f].sum()) for f in FEATURES},
            }
    out["_meta"] = {"thresholds": THRESH, "features": FEATURES}
    return out


def main(weather_path: Path = WEATHER) -> dict:
    rid = pd.read_csv(RIDERSHIP, parse_dates=["date"])
    w = pd.read_csv(weather_path)
    df = add_flags(attach_weather(build_ratio(rid), w))
    res = fit(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    res = main()
    for dt_name in ("weekday", "weekend"):
        print(f"\n[{dt_name}]")
        for band, v in res.get(dt_name, {}).items():
            print(f"  {band:12s} n={v['n']:>5}  {v['effect_pct']}")
