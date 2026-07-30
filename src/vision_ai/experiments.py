"""실험 기록.

모델을 바꿔가며 돌리다 보면 "그때 그 설정이 뭐였지"가 반드시 문제가 된다. 학습 설정과
지표를 같은 줄에 남겨 두면 비교가 되고, 4단계 모델 레지스트리의 토대가 된다.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

RUNS_CSV = "experiments.csv"
RUN_DIR = "runs"

RUN_COLUMNS: tuple[str, ...] = (
    "run_id", "created_at", "kind", "model", "split", "threshold",
    "n_train", "n_eval", "recall", "precision", "f1", "auroc", "average_precision",
    "tp", "fp", "fn", "tn", "config_json", "note",
)

_METRIC_KEYS = (
    "threshold", "recall", "precision", "f1", "auroc", "average_precision",
    "tp", "fp", "fn", "tn",
)


def _runs_path() -> Path:
    return config.ARTIFACT_ROOT / RUNS_CSV


def _run_dir() -> Path:
    return config.ARTIFACT_ROOT / RUN_DIR


def new_run_id() -> str:
    """읽기 쉬운 실행 ID (시간 + 짧은 난수)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def record_run(
    *,
    kind: str,
    metrics: dict,
    settings: dict,
    model: str = "",
    split: str = "",
    n_train: int = 0,
    n_eval: int = 0,
    note: str = "",
    artifacts: dict | None = None,
) -> str:
    """실행 하나를 기록하고 run_id를 반환한다."""
    config.ensure_dirs()
    run_id = new_run_id()

    row = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "model": model,
        "split": split,
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "config_json": json.dumps(settings, ensure_ascii=False, sort_keys=True),
        "note": note,
    }
    for key in _METRIC_KEYS:
        value = metrics.get(key)
        row[key] = float(value) if isinstance(value, (int, float)) else None

    frame = pd.DataFrame([row])[list(RUN_COLUMNS)]
    path = _runs_path()
    header = not path.exists() or path.stat().st_size == 0
    frame.to_csv(path, mode="a", header=header, index=False)

    # 전체 지표는 별도 JSON으로 (CSV 컬럼에 담기 어려운 값 포함)
    detail_dir = _run_dir()
    detail_dir.mkdir(parents=True, exist_ok=True)
    (detail_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "kind": kind,
                "model": model,
                "split": split,
                "settings": settings,
                "metrics": metrics,
                "artifacts": artifacts or {},
                "note": note,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return run_id


def load_runs() -> pd.DataFrame:
    """기록된 실행 목록을 최신순으로 반환한다."""
    path = _runs_path()
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in RUN_COLUMNS})
    df = pd.read_csv(path, dtype={"run_id": "str"})
    for column in RUN_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[list(RUN_COLUMNS)].iloc[::-1].reset_index(drop=True)


def load_run_detail(run_id: str) -> dict | None:
    """실행 상세(JSON)를 읽는다."""
    path = _run_dir() / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_frame() -> pd.DataFrame:
    """실행 비교용 요약 표."""
    runs = load_runs()
    if runs.empty:
        return runs
    columns = [
        "run_id", "created_at", "kind", "model", "n_train", "n_eval",
        "threshold", "recall", "precision", "f1", "auroc", "fn", "fp", "note",
    ]
    return runs[columns]
