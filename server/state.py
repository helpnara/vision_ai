"""화면이 들고 있던 «세션 상태»의 자리.

Streamlit에서 `st.session_state`에 두던 것 중 **다시 계산하면 비싼 것**만 여기 둔다 —
3단계의 마지막 학습 결과(점수 배열·모델 객체), 4단계의 마지막 배치 추론(특징 행렬),
시나리오 결과. 이 앱은 로그인 없는 단일 사용자 도구이고 활성 프로젝트도 프로세스 전역이라
(`config.use_project`) 세션별로 나누는 것이 오히려 일관되지 않다.

프로젝트를 바꾸면 전부 비운다. 다른 프로젝트의 학습 결과가 남아 있으면 평가 리포트가
엉뚱한 데이터를 보여준다.
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_store: dict[str, Any] = {}

KEY_MODEL_RESULT = "p3_result"      # 3단계 마지막 학습·평가 결과 (dict)
KEY_ANOMALY_MODEL = "p3_anomaly"    # 3단계 마지막 이상탐지 모델 객체 (히트맵용)
KEY_DATASET = "p3_dataset"          # 3단계 특징 행렬 캐시 {"signature", "dataset"}
KEY_BATCH = "p4_batch"              # 4단계 마지막 배치 추론 {"version","records","features"}
KEY_SCENARIO = "p4_scenario"        # 4단계 시나리오 결과 객체


def get(key: str, default: Any = None) -> Any:
    with _lock:
        return _store.get(key, default)


def put(key: str, value: Any) -> None:
    with _lock:
        _store[key] = value


def pop(key: str) -> Any:
    with _lock:
        return _store.pop(key, None)


def clear() -> None:
    with _lock:
        _store.clear()
