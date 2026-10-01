"""라우터가 공통으로 쓰는 변환기.

pandas가 돌려주는 값은 그대로 JSON이 되지 않는다 — NaN은 JSON에 없고, numpy 정수는
`json.dumps`가 거부한다. 라우터마다 따로 처리하면 어딘가는 빠뜨리므로 한 곳에서 한다.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException


def jsonable(value: Any) -> Any:
    """numpy·pandas 값을 JSON으로 바꿀 수 있는 파이썬 값으로 푼다. NaN·NaT는 None."""
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.ndarray, list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, pd.Series):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return records(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "__fspath__"):
        return str(value)
    return value


def records(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict]:
    """DataFrame → 행 목록. 열 이름은 문자열로, 값은 JSON 안전하게."""
    if frame is None or frame.empty:
        return []
    view = frame if limit is None else frame.head(limit)
    rows = []
    for row in view.to_dict(orient="records"):
        rows.append({str(key): jsonable(value) for key, value in row.items()})
    return rows


def table(frame: pd.DataFrame, *, limit: int | None = None, index: bool = False) -> dict:
    """화면의 표 컴포넌트가 바로 그릴 수 있는 모양: {columns, rows, total}.

    `index=True`면 인덱스를 첫 열로 넣는다 (교차표처럼 인덱스가 뜻을 가질 때).
    """
    if frame is None or frame.empty:
        return {"columns": [], "rows": [], "total": 0}
    view = frame
    if index:
        view = frame.reset_index()
        # MultiIndex를 풀면 열 이름이 튜플이 될 수 있다
        view.columns = [
            " / ".join(str(part) for part in col) if isinstance(col, tuple) else str(col)
            for col in view.columns
        ]
    else:
        view = view.copy()
        view.columns = [str(col) for col in view.columns]
    return {
        "columns": [str(col) for col in view.columns],
        "rows": records(view, limit=limit),
        "total": int(len(frame)),
    }


def counts(series: pd.Series, *, limit: int | None = None) -> list[dict]:
    """value_counts → [{name, count}] (가로 막대 차트용)."""
    if series is None or series.empty:
        return []
    vc = series.astype(str).value_counts()
    if limit:
        vc = vc.head(limit)
    return [{"name": str(name), "count": int(count)} for name, count in vc.items()]


def bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=str(message))


def not_found(message: str) -> HTTPException:
    return HTTPException(status_code=404, detail=str(message))
