"""화면에 그릴 Vega-Lite 명세. Streamlit 없이 순수 dict를 만든다.

왜 명세를 서버에서 만드는가 — 차트 높이·줄 순서·색은 **실측으로 뒤집힌 값**이고
(`BAND_HEIGHT` 독스트링), 그 결정은 테스트로 잠겨 있어야 한다. 화면(React)은 받은 명세를
vega-embed로 그리기만 한다.
"""

from __future__ import annotations

from typing import Sequence

BAND_HEIGHT = 260
"""두 줄짜리 비교 타임라인의 **전체** 높이(px).

Vega는 이 값을 그림틀 전체 크기로 보고 축 눈금·축 제목·범례가 먼저 자리를 가져간 뒤 **남은
만큼만** 띠를 그린다. 150px으로 두었더니 남은 높이가 두 줄을 담지 못해 **정답 줄과 모델 줄이
겹쳐 그려졌다** — 어긋난 자리를 보라고 만든 화면인데 정작 두 줄이 한 줄로 포개졌다.
"""

# 색은 뜻에 고정한다. 실행할 때마다 바뀌면 «빨간 게 뭐였지»를 매번 다시 읽어야 한다.
BAND_COLORS = {
    "잡음": "#2e9e5b",          # 정답 구간을 잡았다
    "놓침": "#d64545",          # 정답 구간인데 못 잡았다 — 여기가 다음에 볼 곳이다
    "결함 위 알람": "#3b7dd8",   # 알람이 결함 위에서 울렸다
    "헛알람": "#e08b2f",        # 결함이 아닌데 울렸다
}

LANE_ORDER = ("정답 구간", "모델 알람")
"""위에서 아래로 놓는 순서. `segments.LANE_TRUTH`/`LANE_ALARM`과 같은 값이어야 한다.
안 정하면 이름 순으로 정렬돼 «위가 정답»이라는 설명과 그림이 어긋난다."""


def segment_timeline(bands: Sequence[dict], *, duration: float, width: int = 760) -> dict | None:
    """정답 구간과 모델 알람을 위아래 두 줄로 겹쳐 그린다. 그릴 것이 없으면 None.

    끌어 고르는 기능은 없다 — 여기서 하는 일은 **되짚어 보는 것**이지 라벨링이 아니다.
    """
    if not bands:
        return None
    kinds = [kind for kind in BAND_COLORS if any(b["kind"] == kind for b in bands)]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": width,
        "height": BAND_HEIGHT,
        "data": {"values": list(bands)},
        "mark": {"type": "bar", "height": 26, "cornerRadius": 3},
        "encoding": {
            "x": {
                "field": "start", "type": "quantitative",
                "scale": {"domain": [0, max(float(duration), 0.001)]},
                "axis": {"title": "영상 시각 (초)"},
            },
            "x2": {"field": "end"},
            "y": {
                "field": "lane", "type": "nominal", "axis": {"title": None},
                "sort": [LANE_ORDER[0], LANE_ORDER[1]],
            },
            "color": {
                "field": "kind", "type": "nominal",
                "scale": {"domain": kinds, "range": [BAND_COLORS[k] for k in kinds]},
                "legend": {"title": None, "orient": "bottom"},
            },
            "tooltip": [
                {"field": "kind", "type": "nominal", "title": "구분"},
                {"field": "span", "type": "nominal", "title": "구간"},
            ],
        },
    }


def line_chart(rows: Sequence[dict], *, x: str, ys: Sequence[str], x_title: str | None = None,
               height: int = 260) -> dict | None:
    """여러 계열의 선 그래프 (임계값별 재현율/정밀도, 기간별 추이)."""
    if not rows:
        return None
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container",
        "height": height,
        "data": {"values": list(rows)},
        "transform": [{"fold": list(ys), "as": ["series", "value"]}],
        "mark": {"type": "line", "point": len(rows) <= 60},
        "encoding": {
            "x": {"field": x, "type": "quantitative" if _numeric(rows, x) else "ordinal",
                  "axis": {"title": x_title or x}},
            "y": {"field": "value", "type": "quantitative", "axis": {"title": None}},
            "color": {"field": "series", "type": "nominal", "legend": {"title": None, "orient": "bottom"}},
            "tooltip": [{"field": x}, {"field": "series"}, {"field": "value", "format": ".3f"}],
        },
    }


def _numeric(rows: Sequence[dict], key: str) -> bool:
    return all(isinstance(r.get(key), (int, float)) for r in rows)
