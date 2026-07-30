"""4단계: 모델 레지스트리.

3단계 실험 기록(`experiments.csv`)은 "무엇을 시도했는지"의 기록이고, 레지스트리는
"무엇을 쓰고 있는지"의 기록이다. 둘을 분리해야 롤백 대상을 즉시 찾을 수 있다.

승격 시 모델 파일을 **버전 폴더로 복사**한다. 3단계는 같은 경로에 덮어쓰기 때문에,
복사하지 않으면 다음 학습이 과거 버전을 지워 롤백이 불가능해진다.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, experiments

REGISTRY_CSV = "registry.csv"
REGISTRY_DIR = "registry"

STATUS_CANDIDATE = "candidate"
STATUS_PRODUCTION = "production"
STATUS_ARCHIVED = "archived"
STATUSES = (STATUS_CANDIDATE, STATUS_PRODUCTION, STATUS_ARCHIVED)

STATUS_KO = {
    STATUS_CANDIDATE: "후보",
    STATUS_PRODUCTION: "서비스 중",
    STATUS_ARCHIVED: "보관",
}

REGISTRY_COLUMNS: tuple[str, ...] = (
    "version", "created_at", "promoted_at", "status", "run_id", "kind", "model",
    "artifact", "threshold", "recall", "precision", "f1", "auroc",
    "n_train", "n_eval", "note",
)

_METRIC_KEYS = ("threshold", "recall", "precision", "f1", "auroc")


def _registry_path() -> Path:
    return config.ARTIFACT_ROOT / REGISTRY_CSV


def _version_dir(version: str) -> Path:
    return config.ARTIFACT_ROOT / REGISTRY_DIR / version


def empty_registry() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in REGISTRY_COLUMNS})


def load_registry() -> pd.DataFrame:
    """등록된 모델 버전을 최신순으로 반환한다."""
    path = _registry_path()
    if not path.exists():
        return empty_registry()
    df = pd.read_csv(path, dtype={"version": "str", "run_id": "str"})
    for column in REGISTRY_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    for column in _METRIC_KEYS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # 전부 결측인 열은 float64로 추론되어 문자열 대입이 거부된다 (promoted_at 승격 시)
    for column in ("promoted_at", "artifact", "note", "status", "kind", "model"):
        df[column] = df[column].astype("object")
    return df[list(REGISTRY_COLUMNS)].iloc[::-1].reset_index(drop=True)


def _save_registry(df: pd.DataFrame) -> None:
    config.ensure_dirs()
    # 저장은 등록 순서(오래된 것부터)로 되돌려 둔다
    df.iloc[::-1].reset_index(drop=True).to_csv(_registry_path(), index=False)


def next_version(registry: pd.DataFrame | None = None) -> str:
    """다음 버전 번호 (v001, v002, ...)."""
    registry = load_registry() if registry is None else registry
    if registry.empty:
        return "v001"
    numbers = [
        int(str(v)[1:]) for v in registry["version"].astype(str) if str(v).startswith("v") and str(v)[1:].isdigit()
    ]
    return f"v{max(numbers, default=0) + 1:03d}"


# --- 드리프트 기준선 --------------------------------------------------------

def make_baseline(
    X: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    *,
    scores: np.ndarray | None = None,
    quality: pd.DataFrame | None = None,
    label_ratio: float | None = None,
    bins: int = 10,
) -> dict:
    """드리프트 비교용 기준선을 만든다.

    특징별 분위 경계를 저장해 두면, 나중에 들어온 데이터를 같은 경계로 나눠
    분포 이동(PSI)을 계산할 수 있다.
    """
    X = np.asarray(X, dtype=float)
    quantiles = np.linspace(0, 100, bins + 1)
    edges: dict[str, list[float]] = {}
    props: dict[str, list[float]] = {}
    stats: dict[str, dict] = {}

    for index, name in enumerate(feature_names):
        column = X[:, index]
        unique_edges = np.unique(np.percentile(column, quantiles))
        edges[name] = [float(v) for v in unique_edges]

        # 기준 구간의 **실제** 비율을 저장한다. 값에 동점이 많으면 분위 경계가 겹쳐
        # 구간 수가 줄고, 남은 구간이 각각 1/구간수를 담지 않는다. 균등하다고 가정하면
        # 분포가 그대로인데도 PSI가 크게 나온다.
        if unique_edges.size >= 3:
            bounded = unique_edges.copy()
            bounded[0], bounded[-1] = -np.inf, np.inf
            counts, _ = np.histogram(column, bins=bounded)
            total = counts.sum()
            props[name] = [float(c / total) for c in counts] if total else []
        else:
            props[name] = []

        stats[name] = {"mean": float(column.mean()), "std": float(column.std())}

    baseline: dict = {
        "n_samples": int(X.shape[0]),
        "bins": bins,
        "feature_names": list(feature_names),
        "edges": edges,
        "props": props,
        "stats": stats,
    }
    if scores is not None and len(scores):
        scores = np.asarray(scores, dtype=float)
        baseline["score"] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "percentiles": {str(q): float(np.percentile(scores, q)) for q in (5, 25, 50, 75, 95)},
        }
    if quality is not None and not quality.empty:
        baseline["quality"] = {
            column: {
                "mean": float(pd.to_numeric(quality[column], errors="coerce").mean()),
                "std": float(pd.to_numeric(quality[column], errors="coerce").std()),
            }
            for column in ("blur_score", "brightness")
            if column in quality.columns
        }
    if label_ratio is not None:
        baseline["label_ratio"] = float(label_ratio)
    return baseline


def load_baseline(version: str) -> dict | None:
    """버전의 드리프트 기준선을 읽는다."""
    path = _version_dir(version) / "baseline.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --- 등록 / 상태 전이 -------------------------------------------------------

@dataclass
class RegisterResult:
    version: str
    artifact: str | None
    warnings: list[str]


def register(
    run_id: str,
    *,
    artifact: Path | str | None = None,
    baseline: dict | None = None,
    note: str = "",
    promote_now: bool = False,
) -> RegisterResult:
    """3단계 실행을 모델 버전으로 등록한다.

    모델 파일은 버전 폴더로 복사한다 — 3단계가 같은 경로에 덮어쓰므로 복사하지 않으면
    다음 학습이 이 버전을 지운다.
    """
    detail = experiments.load_run_detail(run_id)
    if detail is None:
        raise ValueError(f"실행 기록을 찾을 수 없습니다: {run_id}")

    registry = load_registry()
    version = next_version(registry)
    directory = _version_dir(version)
    directory.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    stored_artifact: str | None = None
    source = Path(artifact) if artifact else None
    if source is None:
        recorded = (detail.get("artifacts") or {}).get("model")
        source = Path(recorded) if recorded else None
    if source is not None and source.exists():
        destination = directory / f"model{source.suffix}"
        shutil.copy2(source, destination)
        stored_artifact = str(destination)
    else:
        warnings.append(
            "모델 파일을 찾을 수 없어 지표만 등록했습니다. 이 버전으로는 추론할 수 없습니다."
        )

    if baseline is not None:
        (directory / "baseline.json").write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        warnings.append("드리프트 기준선이 없어 데이터 드리프트 감시를 할 수 없습니다.")

    (directory / "run.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    metrics = detail.get("metrics") or {}
    row = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "promoted_at": pd.NA,
        "status": STATUS_CANDIDATE,
        "run_id": run_id,
        "kind": detail.get("kind", ""),
        "model": detail.get("model", ""),
        "artifact": stored_artifact,
        "n_train": detail.get("settings", {}).get("n_train", 0) or 0,
        "n_eval": 0,
        "note": note,
    }
    for key in _METRIC_KEYS:
        value = metrics.get(key)
        row[key] = float(value) if isinstance(value, (int, float)) else None

    merged = pd.concat([registry.iloc[::-1], pd.DataFrame([row])], ignore_index=True)
    _save_registry(merged.iloc[::-1].reset_index(drop=True))

    if promote_now:
        promote(version)
    return RegisterResult(version=version, artifact=stored_artifact, warnings=warnings)


def promote(version: str) -> None:
    """버전을 서비스 중으로 올린다. 기존 서비스 버전은 보관으로 내린다."""
    registry = load_registry()
    if version not in set(registry["version"].astype(str)):
        raise ValueError(f"등록되지 않은 버전입니다: {version}")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    versions = registry["version"].astype(str)
    registry.loc[registry["status"] == STATUS_PRODUCTION, "status"] = STATUS_ARCHIVED
    registry.loc[versions == version, "status"] = STATUS_PRODUCTION
    registry.loc[versions == version, "promoted_at"] = now
    _save_registry(registry)


def archive(version: str) -> None:
    """버전을 보관 처리한다."""
    registry = load_registry()
    versions = registry["version"].astype(str)
    if version not in set(versions):
        raise ValueError(f"등록되지 않은 버전입니다: {version}")
    registry.loc[versions == version, "status"] = STATUS_ARCHIVED
    _save_registry(registry)


def production() -> pd.Series | None:
    """현재 서비스 중인 버전. 없으면 None."""
    registry = load_registry()
    if registry.empty:
        return None
    rows = registry[registry["status"] == STATUS_PRODUCTION]
    return None if rows.empty else rows.iloc[0]


def get(version: str) -> pd.Series | None:
    """버전 하나를 조회한다."""
    registry = load_registry()
    if registry.empty:
        return None
    rows = registry[registry["version"].astype(str) == version]
    return None if rows.empty else rows.iloc[0]


def load_run(version: str) -> dict | None:
    """버전에 붙은 3단계 실행 상세를 읽는다."""
    path = _version_dir(version) / "run.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
