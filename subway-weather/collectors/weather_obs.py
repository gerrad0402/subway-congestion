"""서울 ASOS 시간별 관측자료 수집 (kma_sfctm3.php).

- tm1~tm2 는 YYYYMMDDHHmm, 한 번에 최대 31일이라 30일 단위로 나눠 호출합니다.
- 결측값은 -9 / -9.0 으로 오므로 NaN 처리합니다.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from collectors.kma_client import fetch_text, parse_table

SEOUL_STN = 108  # 서울 관측소
RAW_DIR = Path("data/raw")
KEEP = {  # 필요한 컬럼만 남기고 이름 정리
    "YYMMDDHHMI": "obs_time",
    "TA": "temp",       # 기온(°C)
    "RN": "rain_mm",    # 강수량(mm)
    "SD_HR3": "snow_3h",  # 3시간 신적설(cm)
    "SD_TOT": "snow_depth",  # 적설(cm)
    "WS": "wind",       # 풍속(m/s)
    "HM": "humidity",   # 습도(%)
}


def fetch_hourly(start: datetime, end: datetime, stn: int = SEOUL_STN) -> pd.DataFrame:
    frames = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=30), end)
        text = fetch_text(
            "kma_sfctm3.php",
            tm1=cur.strftime("%Y%m%d%H%M"),
            tm2=nxt.strftime("%Y%m%d%H%M"),
            stn=stn,
            help=1,
        )
        frames.append(parse_table(text))
        cur = nxt + timedelta(hours=1)
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEEP if c in df.columns]
    out = df[cols].rename(columns=KEEP).copy()
    out["obs_time"] = pd.to_datetime(out["obs_time"], format="%Y%m%d%H%M")
    for c in out.columns.drop("obs_time"):
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([-9, -9.0], np.nan)
    return out.sort_values("obs_time").reset_index(drop=True)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # 승하차 데이터(2023-01-01 ~ 2025-12-31)와 같은 기간. 마지막 시간대(23~24시)용으로 다음날 00시까지 받음
    raw = fetch_hourly(datetime(2023, 1, 1, 0, 0), datetime(2026, 1, 1, 0, 0))
    raw.to_csv(RAW_DIR / "asos_seoul_raw.csv", index=False)
    clean(raw).to_csv(RAW_DIR / "asos_seoul_clean.csv", index=False)
    print("saved", len(raw), "rows")
