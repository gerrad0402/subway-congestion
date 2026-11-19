"""데이터 로딩 · 학습 · 평가 · 예측 (혼잡도 + 기상 데이터 지원)"""
import io
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    import lightgbm as lgb
except ImportError:  # LightGBM이 없으면 sklearn 모델로 대체
    lgb = None
    from sklearn.ensemble import HistGradientBoostingRegressor

TIME_RE = re.compile(r"^(\d{1,2})시\s*(\d{2})분$")
ID_COLS = ["요일구분", "호선", "역번호", "출발역", "상하구분"]
CATS = ["line", "direction", "daytype"]
WEATHER_COLS = ["temp_c", "precip_mm", "wind_ms", "humidity"]
FEATURES = ["line", "station_no", "direction", "daytype", "minutes"] + WEATHER_COLS
KEY = ["line", "direction", "daytype", "minutes"]

# 혼잡 4단계 경계(%): 서울교통공사 기준을 참고한 값, 필요하면 조정
LEVEL_BINS = [34, 80, 130]
LEVEL_NAMES = ["여유", "보통", "주의", "혼잡"]


def level_name(v: float) -> str:
    return LEVEL_NAMES[int(np.digitize(v, LEVEL_BINS))]


def fmt_time(minutes: int) -> str:
    return f"{int(minutes) // 60 % 24:02d}:{int(minutes) % 60:02d}"


def _slot_to_minutes(s: str) -> int:
    m = TIME_RE.match(str(s).strip())
    if not m:
        raise ValueError(f"시간 형식을 해석할 수 없습니다: {s}")
    h, mm = map(int, m.groups())
    return (h + (24 if h < 5 else 0)) * 60 + mm


# ---------------------------------------------------------------- 데이터
def load_congestion(src) -> pd.DataFrame:
    """가로형 원본 CSV 또는 세로형(혼잡도+기상 병합) CSV를 표준 DataFrame으로 변환.

    열: line, station, station_no, direction, daytype, minutes, congestion,
        temp_c, precip_mm, wind_ms, humidity
    """
    raw = src.read() if hasattr(src, "read") else Path(src).read_bytes()
    df = None
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError("CSV 인코딩을 읽지 못했습니다 (UTF-8, CP949 시도).")

    df.columns = [str(c).strip() for c in df.columns]
    alias = {
        "역명": "출발역",
        "상하선구분": "상하구분",
        "상하선": "상하구분",
        "요일 구분": "요일구분",
        "구분": "요일구분",
        "혼잡도": "congestion",
        "기온(°C)": "temp_c",
        "강수량(mm)": "precip_mm",
        "풍속(m/s)": "wind_ms",
        "습도(%)": "humidity",
    }
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns and v not in df.columns})

    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}\n현재 열: {list(df.columns)[:10]}…")

    tcols = [c for c in df.columns if TIME_RE.match(c)]
    if tcols:
        long = df.melt(id_vars=ID_COLS, value_vars=tcols, var_name="slot", value_name="congestion")
        long["minutes"] = long["slot"].map(_slot_to_minutes)
    elif {"시각", "congestion"}.issubset(df.columns):
        keep_weather = [c for c in WEATHER_COLS if c in df.columns]
        long = df[ID_COLS + ["시각", "congestion"] + keep_weather].copy()
        long["minutes"] = long["시각"].map(_slot_to_minutes)
    else:
        raise ValueError("'5시30분' 형태의 시간대 열 또는 ('시각', '혼잡도') 열을 찾지 못했습니다.")

    long["congestion"] = pd.to_numeric(long["congestion"], errors="coerce")
    long["station_no"] = pd.to_numeric(long["역번호"], errors="coerce")
    long["line"] = long["호선"].astype(str).str.replace("호선", "", regex=False).str.strip()
    long["station"] = long["출발역"].astype(str).str.strip()
    long["direction"] = long["상하구분"].astype(str).str.strip()
    long["daytype"] = long["요일구분"].astype(str).str.strip()

    defaults = {"temp_c": 15.0, "precip_mm": 0.0, "wind_ms": 2.0, "humidity": 60.0}
    for col, default_val in defaults.items():
        if col not in long.columns:
            long[col] = default_val
        else:
            long[col] = pd.to_numeric(long[col], errors="coerce")
            if col == "precip_mm":
                long[col] = long[col].fillna(0.0)
            else:
                hourly_mean = long.groupby("minutes")[col].transform("mean")
                long[col] = long[col].fillna(hourly_mean).fillna(default_val)

    long = long.dropna(subset=["congestion", "station_no"])
    out_cols = ["line", "station", "station_no", "direction", "daytype", "minutes", "congestion"] + WEATHER_COLS
    return long[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------- 학습
def _encoders(df: pd.DataFrame) -> dict:
    return {c: {v: i for i, v in enumerate(sorted(df[c].unique()))} for c in CATS}


def _X(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATS:
        X[c] = X[c].map(enc[c]).fillna(-1).astype(int)
    for c in WEATHER_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X


def _fit(X: pd.DataFrame, y: pd.Series):
    if lgb is not None:
        m = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
        m.fit(X, y, categorical_feature=CATS)
    else:
        mask = [c in CATS for c in X.columns]
        m = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            categorical_features=mask,
            random_state=42,
        )
        m.fit(X, y)
    return m


def evaluate(df: pd.DataFrame, enc: dict, seed: int = 42) -> dict:
    """역 단위 홀드아웃(학습에 없던 역 20%)으로 베이스라인과 비교한다."""
    st = df[["line", "station"]].drop_duplicates().sample(frac=0.2, random_state=seed)
    test_keys = set(st["line"] + "|" + st["station"])
    is_test = (df["line"] + "|" + df["station"]).isin(test_keys)
    tr, te = df[~is_test].copy(), df[is_test].copy()

    base = tr.groupby(KEY)["congestion"].mean().rename("base")
    te = te.join(base, on=KEY)
    te["base"] = te["base"].fillna(tr["congestion"].mean())

    te["pred"] = np.clip(_fit(_X(tr, enc), tr["congestion"]).predict(_X(te, enc)), 0, None)
    yl = np.digitize(te["congestion"], LEVEL_BINS)
    out = {"n_test_stations": len(test_keys), "n_test_rows": len(te)}
    for name, col in [("baseline", "base"), ("model", "pred")]:
        err = te[col] - te["congestion"]
        out[name] = {
            "MAE": float(err.abs().mean()),
            "RMSE": float(np.sqrt((err**2).mean())),
            "F1(4단계)": float(f1_score(yl, np.digitize(te[col], LEVEL_BINS), average="macro")),
        }
    out["resid_q80"] = float((te["pred"] - te["congestion"]).abs().quantile(0.8))
    return out


def train(df: pd.DataFrame) -> dict:
    enc = _encoders(df)
    metrics = evaluate(df, enc)
    model = _fit(_X(df, enc), df["congestion"])
    stations = (
        df.groupby(["line", "station", "station_no"])["direction"]
        .apply(lambda s: sorted(s.unique()))
        .reset_index()
    )
    weather_defaults = {c: float(df[c].mean()) for c in WEATHER_COLS}
    return {
        "model": model,
        "enc": enc,
        "metrics": metrics,
        "stations": stations,
        "daytypes": sorted(df["daytype"].unique()),
        "slots": np.array(sorted(df["minutes"].unique())),
        "weather_defaults": weather_defaults,
        "backend": "LightGBM" if lgb is not None else "sklearn HistGradientBoosting",
    }


def save(bundle: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load(path) -> dict:
    return joblib.load(path)


# ---------------------------------------------------------------- 예측
def predict_curve(
    bundle: dict,
    line: str,
    station: str,
    direction: str,
    daytype: str,
    minutes,
    weather: dict | None = None,
) -> np.ndarray:
    """30분 슬롯 시작 시각에서 예측한 뒤, 요청한 분 단위로 선형 보간한다."""
    st = bundle["stations"]
    row = st[(st["line"] == line) & (st["station"] == station)].iloc[0]
    slots = bundle["slots"]
    w_base = bundle.get("weather_defaults", {"temp_c": 15.0, "precip_mm": 0.0, "wind_ms": 2.0, "humidity": 60.0})
    w_use = {**w_base, **(weather or {})}

    grid = pd.DataFrame({
        "line": line,
        "station_no": row["station_no"],
        "direction": direction,
        "daytype": daytype,
        "minutes": slots,
        **{k: w_use[k] for k in WEATHER_COLS},
    })
    pred = np.clip(bundle["model"].predict(_X(grid, bundle["enc"])), 0, None)
    return np.interp(np.asarray(minutes, dtype=float), slots, pred)
