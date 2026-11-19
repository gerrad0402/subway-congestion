"""기상청 API허브 공통 클라이언트.

- 응답은 EUC-KR 텍스트이고, '#'로 시작하는 줄은 주석/헤더입니다.
- help=1 로 호출하면 컬럼명이 담긴 주석 줄이 함께 옵니다.
"""
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://apihub.kma.go.kr/api/typ01/url"


def _decode(content: bytes) -> str:
    """응답 바이트를 문자열로. 에러 JSON은 UTF-8, 데이터 텍스트는 EUC-KR이라 둘 다 시도한다."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("euc-kr", errors="replace")


def fetch_text(endpoint: str, retries: int = 3, **params) -> str:
    """엔드포인트를 호출하고 EUC-KR 문자열로 디코딩해서 반환.

    4xx(인증/권한/주소 오류)는 재시도해도 소용없으므로 바로 상태 코드와 서버 응답을 보여주며 멈춘다.
    네트워크 오류와 5xx만 재시도한다.
    """
    params["authKey"] = os.environ["KMA_AUTH_KEY"]
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        except requests.RequestException as e:  # 연결 실패, 타임아웃 등
            last_err = e
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            return _decode(r.content)
        body = _decode(r.content)[:300].strip()
        if 400 <= r.status_code < 500:
            hint = ""
            if r.status_code == 403:
                hint = " (이 API의 '활용신청'이 승인되지 않았거나 키에 권한이 없습니다)"
            raise RuntimeError(f"KMA 호출 실패 [{r.status_code}] {endpoint}{hint}\n서버 응답: {body}")
        last_err = RuntimeError(f"HTTP {r.status_code}: {body}")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"KMA 호출 실패(재시도 {retries}회): {endpoint}") from last_err


def parse_table(text: str) -> pd.DataFrame:
    """'#' 주석 줄을 제외하고 공백 구분 테이블을 DataFrame으로 변환.

    컬럼명은 '# YYMMDDHHMI STN WD ...' 처럼 첫 토큰이 시각 컬럼인
    주석 줄에서 가져옵니다.
    """
    columns = None
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            tokens = s.lstrip("#").split()
            # 설명 줄('1. YYMMDDHHMI : ...')과 구분: 첫 토큰이 컬럼명인 줄만 사용
            if tokens and tokens[0] == "YYMMDDHHMI":
                columns = tokens
            continue
        rows.append(s.split())

    if columns is None:
        raise ValueError("컬럼 헤더를 찾지 못했습니다. help=1 로 호출했는지 확인하세요.")
    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns[: len(rows[0])])
