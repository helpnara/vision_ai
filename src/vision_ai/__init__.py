"""비전 기반 표면 결함 탐지 파이프라인 코어 패키지.

단계 구성:
1. ingest   — 데이터 수집(입력)
2. labeling — 라벨링
3. modeling — 모델 개발 및 평가
4. ops      — 사후 운영관리(MLOps): registry · serving · monitoring
"""

from __future__ import annotations

__version__ = "0.4.0"

__all__ = [
    "config", "storage", "datasets", "quality", "ingest", "labeling", "viz",
    "features", "models", "evaluate", "experiments", "claude_review",
    "registry", "serving", "monitoring",
]
