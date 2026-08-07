"""3단계: Claude 비전 2차 판정.

1차 모델(베이스라인·이상탐지)이 애매하다고 본 이미지만 Claude에 넘겨, 결함 유형과
**사람이 읽을 수 있는 판단 근거**를 구조화된 JSON으로 받는다. 전량을 보내지 않는 이유는
호출 비용 때문이고, 근거를 함께 받는 이유는 4단계 판정 이력 조회의 재료가 되기 때문이다.

판정 결과는 `artifacts/claude_reviews.csv`에 캐시한다 — 같은 이미지를 두 번 청구하지 않는다.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, viz

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_EFFORT = "low"  # 단일 이미지 판정은 짧은 과업 — 비용/지연을 낮게 유지한다

# 서버측 폴백: 안전 분류기가 요청을 거절하면 다른 모델로 재시도한다.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

CACHE_PATH = "claude_reviews.csv"

# 결함 유형은 프로젝트 표준(config.DEFECT_TYPES)에 맞춘다 — 자유 서술을 받으면 정규화가 다시 필요해진다.
_TYPE_CHOICES = [*config.DEFECT_TYPES, config.DEFECT_TYPE_NONE]

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "defect": {"type": "boolean"},
        "defect_type": {"type": "string", "enum": _TYPE_CHOICES},
        "severity": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reason": {"type": "string"},
    },
    "required": ["defect", "defect_type", "severity", "confidence", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "당신은 제조 공정의 표면 결함 검사 보조자입니다. 제시된 이미지에서 표면 결함이 있는지 "
    "판정하고, 있다면 유형과 근거를 답합니다.\n\n"
    "판정 원칙:\n"
    "- 결함을 놓치는 것(미탐)이 잘못 잡는 것(오탐)보다 리스크가 큽니다. 애매하면 결함 가능성을 "
    "  낮은 확신도로 보고하되, 근거 없이 결함이라고 단정하지는 마십시오.\n"
    "- 조명 반사, 배경, 초점 흐림은 결함이 아닙니다. 대상 물체 표면 자체의 이상만 결함입니다.\n"
    "- reason에는 무엇을 보고 그렇게 판단했는지 이미지에서 관찰한 근거를 한국어 두 문장 이내로 "
    "  적으십시오. 관찰하지 않은 내용을 추측해 쓰지 마십시오.\n"
    "- 결함이 없으면 defect=false, defect_type=\"none\", severity=1로 답하십시오."
)


@dataclass
class ReviewResult:
    """Claude 2차 판정 결과."""

    image_id: str
    defect: bool
    defect_type: str
    severity: int
    confidence: str
    reason: str
    model: str
    status: str = "ok"          # ok | refusal | error | unavailable
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _failed(image_id: str, status: str, reason: str, model: str = DEFAULT_MODEL) -> ReviewResult:
    return ReviewResult(
        image_id=image_id,
        defect=False,
        defect_type=config.DEFECT_TYPE_NONE,
        severity=1,
        confidence="low",
        reason=reason,
        model=model,
        status=status,
    )


# --- 사용 가능 여부 --------------------------------------------------------

def sdk_installed() -> bool:
    """anthropic SDK가 설치되어 있는지."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def credentials_hint() -> str | None:
    """자격 증명이 없어 보일 때 안내 문구를 반환한다. 확실하면 None.

    환경변수가 비어 있어도 `ant auth login` 프로필로 인증될 수 있으므로, 단정하지 않고
    힌트만 준다.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return None
    return (
        "`ANTHROPIC_API_KEY` 환경변수가 설정되어 있지 않습니다. "
        "`ant auth login`으로 로그인한 프로필이 있으면 그대로 동작하고, 없으면 키가 필요합니다."
    )


def is_available() -> bool:
    """2차 판정을 시도할 수 있는 상태인지 (SDK 설치 여부만 확인)."""
    return sdk_installed()


def _client():
    import anthropic

    return anthropic.Anthropic()


# --- 이미지 준비 -----------------------------------------------------------

def prepare_image(
    rgb: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    *,
    margin: int = 24,
    max_side: int = 1024,
) -> bytes:
    """판정에 보낼 PNG 바이트를 만든다. ROI가 있으면 여유를 두고 잘라낸다.

    결함은 국소적이라 전체 이미지를 그대로 보내면 결함이 몇 픽셀로 축소된다.
    ROI 크롭이 판정 품질에 직접 영향을 준다.
    """
    import cv2

    image = viz.crop(rgb, roi, margin=margin) if roi else rgb
    longest = max(image.shape[:2])
    if longest > max_side:
        scale = max_side / longest
        image = cv2.resize(
            image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("이미지를 PNG로 인코딩할 수 없습니다.")
    return buffer.tobytes()


def _build_prompt(
    category: str | None, candidate_type: str | None, model_score: float | None
) -> str:
    lines = ["이 이미지에 표면 결함이 있는지 판정하십시오."]
    if category:
        lines.append(f"대상 물체 종류: {category}")
    if candidate_type and candidate_type not in (config.DEFECT_TYPE_NONE, config.DEFECT_TYPE_UNSPECIFIED):
        lines.append(
            f"1차 모델이 추정한 결함 유형: {candidate_type} "
            "(참고용입니다. 동의하지 않으면 다르게 판정하십시오.)"
        )
    if model_score is not None:
        lines.append(f"1차 모델 결함 점수: {model_score:.3f} (높을수록 결함 의심)")
    lines.append(
        "가능한 결함 유형: "
        + ", ".join(f"{k}({v})" for k, v in config.DEFECT_TYPES.items())
    )
    return "\n".join(lines)


# --- 판정 -----------------------------------------------------------------

def review_image(
    rgb: np.ndarray,
    *,
    image_id: str,
    roi: tuple[int, int, int, int] | None = None,
    category: str | None = None,
    candidate_type: str | None = None,
    model_score: float | None = None,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    use_fallback: bool = True,
) -> ReviewResult:
    """이미지 1장을 Claude에 보내 판정을 받는다. 실패해도 예외를 던지지 않는다."""
    if not sdk_installed():
        return _failed(
            image_id, "unavailable",
            "anthropic SDK가 설치되지 않았습니다 (`pip install anthropic`).", model,
        )

    import anthropic

    try:
        png = prepare_image(rgb, roi)
    except ValueError as exc:
        return _failed(image_id, "error", f"이미지 준비 실패: {exc}", model)

    request = {
        "model": model,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "output_config": {
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            "effort": effort,
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.standard_b64encode(png).decode("utf-8"),
                        },
                    },
                    {"type": "text", "text": _build_prompt(category, candidate_type, model_score)},
                ],
            }
        ],
    }

    try:
        client = _client()
    except Exception as exc:  # 자격 증명 해석 실패 등
        return _failed(image_id, "unavailable", f"클라이언트 생성 실패: {exc}", model)

    try:
        response = _send(client, request, use_fallback=use_fallback)
    except anthropic.AuthenticationError:
        return _failed(image_id, "unavailable", "인증 실패 — API 키 또는 로그인 프로필을 확인하세요.", model)
    except anthropic.RateLimitError:
        return _failed(image_id, "error", "요청 한도 초과 — 잠시 후 다시 시도하세요.", model)
    except anthropic.APIStatusError as exc:
        return _failed(image_id, "error", f"API 오류 {exc.status_code}: {exc.message}", model)
    except anthropic.APIConnectionError:
        return _failed(image_id, "error", "네트워크 오류 — 연결을 확인하세요.", model)

    # 안전 분류기가 거절하면 content가 비어 있거나 부분 응답이다 — 먼저 확인해야 한다.
    if getattr(response, "stop_reason", None) == "refusal":
        return _failed(image_id, "refusal", "모델이 판정을 거절했습니다.", getattr(response, "model", model))

    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
    if not text:
        return _failed(image_id, "error", "응답에 텍스트가 없습니다.", getattr(response, "model", model))

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _failed(image_id, "error", f"JSON 파싱 실패: {text[:200]}", getattr(response, "model", model))

    usage = getattr(response, "usage", None)
    return ReviewResult(
        image_id=image_id,
        defect=bool(payload["defect"]),
        defect_type=str(payload["defect_type"]),
        severity=int(payload["severity"]),
        confidence=str(payload["confidence"]),
        reason=str(payload["reason"]),
        model=str(getattr(response, "model", model)),
        status="ok",
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


def _send(client, request: dict, *, use_fallback: bool):
    """요청을 보낸다. 서버측 폴백을 우선 시도하고, 거부되면 폴백 없이 재시도한다."""
    import anthropic

    if use_fallback:
        try:
            return client.beta.messages.create(
                **request, betas=[_FALLBACK_BETA], fallbacks="default"
            )
        except anthropic.BadRequestError as exc:
            # 이 계정/환경에서 폴백 베타를 못 쓰는 경우 — 기능 자체를 막지 않는다
            message = str(getattr(exc, "message", exc)).lower()
            if "fallback" not in message and "beta" not in message:
                raise
    return client.messages.create(**request)


# --- 결과 캐시 -------------------------------------------------------------

REVIEW_COLUMNS: tuple[str, ...] = (
    "image_id", "defect", "defect_type", "severity", "confidence", "reason",
    "model", "status", "input_tokens", "output_tokens", "reviewed_at",
)


def _cache_path() -> Path:
    return config.artifact_root() / CACHE_PATH


def load_reviews() -> pd.DataFrame:
    """저장된 2차 판정 결과를 읽는다."""
    path = _cache_path()
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in REVIEW_COLUMNS})
    df = pd.read_csv(path, dtype={"image_id": "str"})
    for column in REVIEW_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[list(REVIEW_COLUMNS)]


def save_reviews(results: list[ReviewResult]) -> int:
    """판정 결과를 캐시에 추가한다 (같은 image_id는 최신 값으로 대체)."""
    if not results:
        return 0
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    incoming = pd.DataFrame([{**r.to_dict(), "reviewed_at": now} for r in results])
    incoming = incoming[list(REVIEW_COLUMNS)]

    current = load_reviews()
    merged = (
        pd.concat([current, incoming], ignore_index=True)
        .drop_duplicates(subset="image_id", keep="last")
        .reset_index(drop=True)
    )
    config.ensure_dirs()
    merged.to_csv(_cache_path(), index=False)
    return len(incoming)


def reviewed_ids() -> set[str]:
    """이미 판정한 image_id 집합 — 중복 청구를 막는다."""
    df = load_reviews()
    if df.empty:
        return set()
    return set(df.loc[df["status"] == "ok", "image_id"].astype(str))


def select_uncertain(
    errors: pd.DataFrame, *, threshold: float, band: float = 0.15, limit: int = 20
) -> pd.DataFrame:
    """판정 임계값 근처의 '애매한' 이미지를 골라낸다.

    전량을 Claude에 보내면 비용이 선형으로 늘어난다. 1차 모델이 확신하지 못한 구간만
    2차 판정에 넘기는 것이 하이브리드 구성의 핵심이다.
    """
    if errors.empty:
        return errors
    scores = errors["score"].astype(float)
    span = float(scores.max() - scores.min()) or 1.0
    distance = (scores - threshold).abs() / span
    picked = errors.assign(uncertainty=1.0 - distance)
    return picked.sort_values("uncertainty", ascending=False).head(limit).reset_index(drop=True)


def estimate_cost(count: int, *, input_tokens: int = 1800, output_tokens: int = 200) -> float:
    """호출 비용을 대략 추정한다 (USD).

    claude-opus-5 기준 입력 $5 / 출력 $25 per MTok. 이미지 토큰은 해상도에 따라 달라지므로
    어디까지나 참고값이다.
    """
    return count * (input_tokens * 5.0 + output_tokens * 25.0) / 1_000_000
