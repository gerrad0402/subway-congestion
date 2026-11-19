"""대중교통 혼잡도 예측 · 덜 붐비는 출발 시간 추천 (Streamlit 데모)

실행: streamlit run app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import model as M
import recommend as R

DATA_DIRS = [Path("data/processed"), Path("data/congestion")]
MODEL_PATH = Path("models/model.joblib")

st.set_page_config(page_title="혼잡도 기반 출발 시간 추천", layout="wide")


def _find_csv() -> Path | None:
    for d in DATA_DIRS:
        if d.exists():
            csvs = sorted(d.glob("*.csv"))
            if csvs:
                return csvs[0]
    return None


@st.cache_resource(show_spinner="모델을 학습하는 중입니다…")
def get_bundle():
    if MODEL_PATH.exists():
        return M.load(MODEL_PATH)
    csv_path = _find_csv()
    if csv_path is None:
        return None
    bundle = M.train(M.load_congestion(csv_path))
    M.save(bundle, MODEL_PATH)
    return bundle


st.title("덜 붐비는 출발 시간 추천")
st.caption("서울 지하철 혼잡도 + 기상 데이터 기반 · 조회 → 예측 → 출발 시간 추천")

bundle = get_bundle()

if bundle is None:
    st.info("`data/processed/` 또는 `data/congestion/` 폴더에 CSV를 넣거나, 아래에서 업로드하세요.")
    up = st.file_uploader("혼잡도 CSV 업로드", type="csv")
    if up is not None:
        try:
            with st.spinner("학습 중…"):
                bundle = M.train(M.load_congestion(up))
                M.save(bundle, MODEL_PATH)
            get_bundle.clear()
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    st.stop()

stations = bundle["stations"]
w_def = bundle.get("weather_defaults", {"temp_c": 15.0, "precip_mm": 0.0, "wind_ms": 2.0, "humidity": 60.0})

with st.sidebar:
    st.header("조건 입력")
    daytype = st.selectbox("요일 구분", bundle["daytypes"])
    desired = st.time_input("희망 출발 시간", value=pd.Timestamp("08:00").time(), step=600)
    desired_min = desired.hour * 60 + desired.minute
    if desired_min < 330:
        desired_min += 24 * 60
    window = st.select_slider("탐색 범위 (± 분)", [30, 60, 90, 120], value=60)
    penalty = st.slider(
        "일찍·늦게 가는 불편 (10분당 %p)",
        0.0,
        5.0,
        0.5,
        0.1,
        help="값이 클수록 희망 시간에 가까운 시간을 추천합니다.",
    )
    agg = st.radio("경로 혼잡도 계산", ["가장 붐비는 구간", "구간 평균"], horizontal=True)

    st.divider()
    st.subheader("기상 조건")
    temp_c = st.slider("기온 (°C)", -15.0, 40.0, float(round(w_def["temp_c"], 1)), 0.5)
    precip_mm = st.slider("강수량 (mm)", 0.0, 50.0, float(round(w_def["precip_mm"], 1)), 0.5)
    wind_ms = st.slider("풍속 (m/s)", 0.0, 20.0, float(round(w_def["wind_ms"], 1)), 0.5)
    humidity = st.slider("습도 (%)", 10.0, 100.0, float(round(w_def["humidity"], 1)), 1.0)
    weather = {"temp_c": temp_c, "precip_mm": precip_mm, "wind_ms": wind_ms, "humidity": humidity}

    st.divider()
    n_legs = st.radio("이용 구간 수", [1, 2], horizontal=True, help="2구간은 환승 경로입니다.")
    legs = []
    for i in range(n_legs):
        st.subheader(f"구간 {i + 1}")
        line = st.selectbox(
            "호선",
            sorted(stations["line"].unique(), key=lambda x: (len(x), x)),
            key=f"l{i}",
        )
        sub = stations[stations["line"] == line].sort_values("station_no")
        name = st.selectbox("탑승역", sub["station"].tolist(), key=f"s{i}")
        dirs = sub[sub["station"] == name].iloc[0]["direction"]
        direction = st.selectbox("방향", dirs, key=f"d{i}")
        ride = 0
        if i == 1:
            ride = st.number_input("구간 1 탑승 후 이 역까지 걸리는 시간 (분)", 0, 120, 20, key="ride")
        legs.append(R.Leg(line, name, direction, int(ride)))

result = R.recommend(bundle, legs, daytype, desired_min, window, penalty, agg, weather=weather)
now, best = result["desired"], result["best"]

c1, c2, c3 = st.columns(3)
c1.metric(
    "희망 시간 혼잡도",
    f"{now['score']:.0f}%",
    f"{M.fmt_time(now['t'])} · {M.level_name(now['score'])}",
    delta_color="off",
)
c2.metric("추천 출발 시간", M.fmt_time(best["t"]), f"{best['t'] - desired_min:+d}분", delta_color="off")
c3.metric(
    "추천 시간 혼잡도",
    f"{best['score']:.0f}%",
    f"{best['score'] - now['score']:+.0f}%p · {M.level_name(best['score'])}",
    delta_color="inverse",
)

if best["t"] == desired_min:
    st.success("희망 시간이 탐색 범위에서 가장 한산해요. 그대로 출발하셔도 좋습니다.")
else:
    st.success(
        f"{M.fmt_time(best['t'])}에 출발하면 {M.fmt_time(desired_min)}보다 혼잡도가 "
        f"{now['score'] - best['score']:.0f}%p 낮아요."
    )

curve = result["curve"]
curve["시각"] = curve["t"].map(M.fmt_time)
x = alt.X("시각:O", sort=None, title="출발 시간", axis=alt.Axis(labelAngle=0))
band = alt.Chart(curve).mark_area(opacity=0.2).encode(x=x, y="low:Q", y2="high:Q")
line_ = alt.Chart(curve).mark_line(point=True).encode(
    x=x,
    y=alt.Y("score:Q", title="예측 혼잡도 (%)"),
    tooltip=["시각", alt.Tooltip("score:Q", format=".0f")],
)
rules = alt.Chart(
    pd.DataFrame({"시각": [M.fmt_time(desired_min), M.fmt_time(best["t"])], "구분": ["희망", "추천"]})
).mark_rule(strokeDash=[4, 4]).encode(x=x, color=alt.Color("구분:N", legend=alt.Legend(title=None)))
st.altair_chart((band + line_ + rules).properties(height=360), use_container_width=True)
st.caption(f"음영은 학습에 없던 역에서 측정한 오차(80% 범위 ±{bundle['metrics']['resid_q80']:.0f}%p)를 반영한 예측 구간입니다.")

with st.expander("후보 시간 비교"):
    show = curve[["시각", "score", "low", "high"]].rename(
        columns={"score": "예측(%)", "low": "하한(%)", "high": "상한(%)"}
    ).round(0)
    show["등급"] = curve["score"].map(M.level_name)
    st.dataframe(show, hide_index=True, use_container_width=True)

with st.expander("모델 성능 (학습에 없던 역 기준)"):
    m = bundle["metrics"]
    st.write(f"학습 모델: {bundle['backend']} · 평가 역 {m['n_test_stations']}개 / {m['n_test_rows']:,}행")
    st.dataframe(
        pd.DataFrame({"베이스라인(호선·방향·요일·시간대 평균)": m["baseline"], "모델": m["model"]}).round(3).T,
        use_container_width=True,
    )
    if st.button("모델 다시 학습 (CSV 교체 후)"):
        MODEL_PATH.unlink(missing_ok=True)
        get_bundle.clear()
        st.rerun()
