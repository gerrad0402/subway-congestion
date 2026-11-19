"""서울교통공사 역별 시간대별 승하차 CSV(연도별 파일) → 날짜×호선×시간 long 형태.

- 연도별로 열 이름이 달라서('06-07시간대' / '06시-07시' / '06시 이전' ...) 시작 시각만 뽑아 통일합니다.
- 06시 이전 → hour=5, 24시 이후 → hour=24 로 둡니다.
- 역 단위 원본은 60만 행이라 호선 단위로 합산해서 저장합니다.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/processed/ridership_hourly.csv")


def _read(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 읽지 못했습니다: {path}")


def _hour_names(tcols: list[str]) -> list[int]:
    """시간대 열 이름 → 시작 시각(정수). 첫 열은 '06시 이전', 마지막 열은 '24시 이후'."""
    hours = [5]
    for c in tcols[1:-1]:
        hours.append(int(re.match(r"\d+", c).group()))
    hours.append(24)
    return hours


def load_file(path: Path) -> pd.DataFrame:
    df = _read(path)
    # 위치 기준: 날짜(1), 호선(2), 역번호(3), 역명(4), 승하차(5), 이후 시간대
    date_col, line_col, io_col = df.columns[1], df.columns[2], df.columns[5]
    tcols = list(df.columns[6:])
    hours = _hour_names(tcols)
    df = df.rename(columns={date_col: "date", line_col: "line", io_col: "io"})
    df = df.dropna(subset=["date", "line", "io"])
    df["date"] = pd.to_datetime(df["date"])
    df["line"] = df["line"].astype(str).str.replace("호선", "", regex=False).str.strip()
    df[tcols] = df[tcols].apply(pd.to_numeric, errors="coerce")
    long = df.melt(id_vars=["date", "line", "io"], value_vars=tcols,
                   var_name="band", value_name="count")
    long["hour"] = long["band"].map(dict(zip(tcols, hours)))
    return long.groupby(["date", "line", "io", "hour"], as_index=False)["count"].sum()


def build(src_dir: str | Path, pattern: str = "*승하차인원*.csv") -> pd.DataFrame:
    files = sorted(Path(src_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"{src_dir} 에서 {pattern} 파일을 찾지 못했습니다.")
    out = pd.concat([load_file(f) for f in files], ignore_index=True)
    # 같은 날짜가 두 파일에 겹치면 마지막 값을 사용
    out = out.drop_duplicates(["date", "line", "io", "hour"], keep="last")
    return out.sort_values(["date", "line", "io", "hour"]).reset_index(drop=True)


def flag_offdays(daily_total: pd.Series, threshold: float = 0.85) -> pd.Series:
    """평일인데 같은 해 같은 요일 중앙값의 85% 미만이면 휴일/연휴로 간주 (공휴일 목록 없이 데이터로 판별)."""
    d = daily_total.to_frame("t")
    d["wd"] = d.index.weekday
    d["ratio"] = d["t"] / d.groupby([d.index.year, "wd"])["t"].transform("median")
    return (d["wd"] < 5) & (d["ratio"] < threshold)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/ridership"
    df = build(src)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df):,} rows  {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")
