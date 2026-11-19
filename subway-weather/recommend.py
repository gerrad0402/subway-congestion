"""출발 시간 추천 로직 (Streamlit과 분리해 단독 테스트 가능)"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import model as M

STEP = 10                      # 후보 출발 시간 간격(분)
DAY_START, DAY_END = 330, 1470 # 운행 시간대: 05:30 ~ 익일 00:30


@dataclass
class Leg:
    line: str
    station: str
    direction: str
    offset: int = 0            # 출발 후 이 구간 역에 도착하기까지 걸리는 분


def recommend(
    bundle: dict,
    legs: list,
    daytype: str,
    desired: int,
    window: int = 60,
    penalty: float = 0.5,
    agg: str = "가장 붐비는 구간",
    weather: dict | None = None,
) -> dict:
    lo, hi = max(DAY_START, desired - window), min(DAY_END, desired + window)
    grid = np.arange(lo, hi + 1, STEP)
    if desired not in grid:
        grid = np.sort(np.append(grid, desired))

    per_leg = np.vstack([
        M.predict_curve(bundle, g.line, g.station, g.direction, daytype, grid + g.offset, weather=weather)
        for g in legs
    ])
    score = per_leg.max(axis=0) if agg == "가장 붐비는 구간" else per_leg.mean(axis=0)
    err = bundle["metrics"]["resid_q80"]
    curve = pd.DataFrame({
        "t": grid,
        "score": score,
        "low": np.clip(score - err, 0, None),
        "high": score + err,
    })

    cost = curve["score"] + penalty * np.abs(curve["t"] - desired) / 10
    best = curve.loc[cost.idxmin()]
    now = curve.loc[curve["t"] == desired].iloc[0]
    now, best = now.to_dict(), best.to_dict()
    now["t"], best["t"] = int(now["t"]), int(best["t"])
    return {"curve": curve.copy(), "desired": now, "best": best}
