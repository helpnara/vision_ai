"""이미지 특징 캐시.

특징 추출은 장당 약 16ms다. 4,584장이면 74초, 라벨을 조금 고치고 다시 학습할 때마다
같은 74초를 다시 쓴다. 세션에만 담아두면 새로고침 한 번에 사라진다.

## 왜 이미지 단위인가

"이번에 쓴 이미지 집합"을 통째로 캐싱하면, 이미지를 10장 추가하는 순간 전체가 무효가 되어
4,594장을 다시 뽑는다. 이미지 하나하나를 열쇠로 삼으면 **새로 들어온 10장만** 뽑으면 된다.
운영 중에는 데이터가 조금씩 늘어나므로 이쪽이 실제 사용 방식에 맞는다.

## 무엇이 바뀌면 다시 뽑아야 하는가

1. **이미지 파일이 바뀌었을 때** — 같은 경로에 다시 촬영한 사진을 덮어쓰는 일이 있다.
   수정 시각과 크기를 함께 저장해 다르면 다시 뽑는다. 내용 해시가 더 정확하지만 파일을
   전부 읽어야 해서, 그럴 바에는 특징을 다시 뽑는 편이 낫다.
2. **특징 정의가 바뀌었을 때** — 특징을 추가하면 예전 벡터는 길이부터 다르다.
   특징 이름 목록에서 지문을 만들어 저장하고, 다르면 캐시를 통째로 버린다.

캐시는 프로젝트별 ``artifacts/projects/<slug>/cache/``에 둔다. 지워도 다시 계산될 뿐 잃는 정보가 없다.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, features

CACHE_NAME = "image_features.npz"


def cache_path() -> Path:
    return config.cache_dir() / CACHE_NAME


def schema_fingerprint() -> str:
    """특징 정의가 바뀌었는지 알아보는 지문.

    이름 목록과 전처리 크기를 함께 넣는다 — 이름이 같아도 입력 크기가 달라지면 값이 달라진다.
    """
    payload = "|".join(features.FEATURE_NAMES) + f"@{features.IMAGE_SIZE}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stamp(path: str | os.PathLike[str]) -> str:
    """파일이 바뀌었는지 알아보는 표식. 읽을 수 없으면 빈 문자열."""
    try:
        info = Path(path).stat()
    except OSError:
        return ""
    return f"{int(info.st_mtime_ns)}:{info.st_size}"


@dataclass
class FeatureCache:
    """image_id → 특징 벡터. 파일이 바뀌지 않은 것만 유효하다."""

    schema: str = field(default_factory=schema_fingerprint)
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    stamps: dict[str, str] = field(default_factory=dict)

    def get(self, image_id: str, path: str | os.PathLike[str]) -> np.ndarray | None:
        """캐시된 벡터. 없거나 파일이 바뀌었으면 None."""
        vector = self.vectors.get(image_id)
        if vector is None:
            return None
        current = stamp(path)
        # 파일을 읽을 수 없으면(빈 표식) 캐시를 믿지 않는다 — 지워졌을 수 있다.
        if not current or self.stamps.get(image_id) != current:
            return None
        return vector

    def put(self, image_id: str, path: str | os.PathLike[str], vector: np.ndarray) -> None:
        current = stamp(path)
        if not current:
            return
        self.vectors[image_id] = np.asarray(vector, dtype=np.float32)
        self.stamps[image_id] = current

    def __len__(self) -> int:
        return len(self.vectors)


def load() -> FeatureCache:
    """캐시를 읽어온다. 없거나 깨졌거나 특징 정의가 바뀌었으면 빈 캐시를 준다."""
    schema = schema_fingerprint()
    path = cache_path()
    if not path.exists():
        return FeatureCache(schema=schema)
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"]) != schema:
                return FeatureCache(schema=schema)
            ids = [str(i) for i in data["image_ids"]]
            marks = [str(s) for s in data["stamps"]]
            matrix = np.asarray(data["X"], dtype=np.float32)
    except (OSError, ValueError, KeyError):
        # 저장 도중 죽어 잘린 파일 등. 캐시는 다시 만들면 되므로 조용히 버린다.
        return FeatureCache(schema=schema)

    if not (len(ids) == len(marks) == matrix.shape[0]):
        return FeatureCache(schema=schema)
    return FeatureCache(
        schema=schema,
        vectors={i: matrix[n] for n, i in enumerate(ids)},
        stamps=dict(zip(ids, marks)),
    )


def save(cache: FeatureCache) -> None:
    """캐시를 저장한다. 실패해도 예외를 올리지 않는다 — 캐시가 없어도 앱은 돌아간다."""
    if not cache.vectors:
        return
    ids = list(cache.vectors)
    matrix = np.stack([cache.vectors[i] for i in ids]).astype(np.float32)
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 같은 캐시를 두 세션이 동시에 쓸 수 있다. 임시 파일에 쓴 뒤 바꿔치기해야
        # 읽는 쪽이 반쯤 쓰인 파일을 만나지 않는다.
        temporary = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temporary,
            schema=np.array(cache.schema),
            image_ids=np.array(ids),
            stamps=np.array([cache.stamps.get(i, "") for i in ids]),
            X=matrix,
        )
        os.replace(temporary, path)
    except OSError:
        return


def clear() -> None:
    """캐시를 지운다. 다음 학습 때 다시 계산된다."""
    try:
        cache_path().unlink(missing_ok=True)
    except OSError:
        pass


def summary() -> dict:
    """화면에 띄울 캐시 현황."""
    path = cache_path()
    if not path.exists():
        return {"count": 0, "size_mb": 0.0}
    cache = load()
    return {"count": len(cache), "size_mb": path.stat().st_size / 1024**2}
