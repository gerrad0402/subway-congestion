"""동작 확인용 가상 데이터 생성 (실제 서울교통공사 CSV와 같은 열 구조). 실제 분석에는 쓰지 마세요."""
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
slots = [f"{h}시{m:02d}분" for h in list(range(5, 24)) + [0] for m in (0, 30) if not (h == 5 and m == 0)]
rows = []
for line, n in [(1, 20), (2, 30), (7, 18)]:
    for no in range(n):
        base = rng.uniform(0.5, 1.3)
        for day in ["평일", "토요일", "일요일"]:
            for d in ["상선", "하선"]:
                vals = []
                for s in slots:
                    h = int(s.split("시")[0]); h = h + 24 if h < 5 else h
                    peak = 90 * np.exp(-((h - 8) ** 2) / 2) * (d == "상선") + 80 * np.exp(-((h - 18.5) ** 2) / 3) * (d == "하선")
                    v = (20 + peak) * base * (1 if day == "평일" else 0.5) + rng.normal(0, 4)
                    vals.append(round(max(v, 2), 1))
                rows.append([day, line, 100 * line + no, f"{line}호선_역{no}", d] + vals)
pd.DataFrame(rows, columns=["요일구분", "호선", "역번호", "출발역", "상하구분"] + slots).to_csv(
    "sample_congestion.csv", index=False, encoding="cp949")
print("sample_congestion.csv 생성")
