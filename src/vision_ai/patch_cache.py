"""이상탐지의 격자(패치) 특징 캐시.

## 왜 분류용 캐시(`feature_cache`)를 그대로 못 쓰는가

**크기가 약 천 배 다르다.** 분류는 이미지 한 장이 숫자 14개(56바이트)지만, 이상탐지는
31×31 격자마다 14개라 장당 53KB다. 4,584장이면 float32로 0.25GB — `feature_cache`가 쓰는
"전부 읽어 사전에 담고 압축해 통째로 다시 쓰기"를 그대로 하면 저장할 때마다 그 0.25GB를
압축하게 된다(측정: 11.3초). 그래서 저장 방식이 달라야 한다.

## 재 보고 정한 것 세 가지

**측정 (실제 VisA 이미지 1,000장, 이 환경)**

| | 시간 |
|---|---|
| 캐시 없이 (이미지 읽기 + 특징 추출) | **42.4초** |
| 캐시 있을 때 | **0.15초** (280배) |

4,584장으로 환산하면 **194초 → 0.7초**, 캐시 크기는 **0.12GB**다.

저장 방식은 따로 재서 골랐다 (4,584장 크기 기준):

| 방식 | 저장 / 읽기 | 크기 |
|---|---|---|
| npz 압축 (float32) | 11.3초 / 1.6초 | 0.22GB |
| npy 무압축 (float32) | 1.5초 / 0.4초 | 0.25GB |
| npy 무압축 (**float16**) | **0.3초 / 0.1초** | **0.12GB** |

1. **캐시를 둔다.** 이상탐지 탭은 설정을 바꿔 가며 여러 번 돌리는 화면인데, 그 194초를
   누를 때마다 다시 쓰고 있었다.
2. **압축하지 않는다.** 압축은 저장을 11.3초로 만들면서 0.03GB만 줄인다. 남는 장사가 아니다.
3. **float16으로 저장한다.** 값 범위가 2~159라 표현 한계(65504)에 한참 못 미친다.
   왕복 상대 오차는 최대 4.9e-4이고, 실제 평가에서 **AUROC는 소수점 여섯 자리까지
   같았다**(0.904167). 디스크가 절반이 되는 값으로 충분히 싸다.

## 왜 배열 하나에 몰아 두고 memmap으로 읽는가

이미지마다 파일을 하나씩 두면 4,584개가 생기고, 파일이 바뀌었는지 볼 표식을 어디 둘지가
또 문제가 된다. 배열 하나(`data.npy`) + 색인 하나(`index.json`)로 두고 **memmap으로 열면
필요한 줄만 디스크에서 읽는다.** 이번 학습에 쓰는 것만 메모리에 올라온다.

새로 뽑은 것이 없으면 **다시 쓰지 않는다.** 두 번째 실행부터는 읽기만 한다.

## 설정이 바뀌면 어떻게 되는가

백엔드(고전 CV / CNN)·패치 크기·간격·입력 크기가 달라지면 격자 자체가 다른 것이라
섞으면 안 된다. 설정에서 만든 지문을 **폴더 이름**으로 쓴다 — 그래서 백엔드를 바꿨다가
되돌려도 앞서 뽑아 둔 것이 그대로 남아 있다.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, features

PATCHES_DIR = "patches"
DATA_NAME = "data.npy"
INDEX_NAME = "index.json"
STORED_DTYPE = np.float16
"""저장 형식. 실측에서 왕복 상대 오차 최대 4.9e-4, AUROC는 여섯 자리까지 동일했다."""


def schema_fingerprint(anomaly_config) -> str:
    """격자 특징의 정의가 같은지 알아보는 지문.

    백엔드·패치 크기·간격·입력 크기가 하나라도 다르면 격자 모양이나 값이 달라진다.
    패치 특징의 평면 목록도 함께 넣는다 — 평면을 추가하면 차원부터 달라진다.
    """
    parts = [
        str(getattr(anomaly_config, "backend", "classic")),
        f"p{getattr(anomaly_config, 'patch', 0)}",
        f"s{getattr(anomaly_config, 'stride', 0)}",
        f"i{getattr(anomaly_config, 'size', features.IMAGE_SIZE)}",
        "|".join(features.PATCH_PLANES) + f"x{features.PATCH_FEATURE_DIM}",
    ]
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:16]


def cache_dir(schema: str) -> Path:
    return config.cache_dir() / PATCHES_DIR / schema


def stamp(path: str | os.PathLike[str]) -> str:
    """파일이 바뀌었는지 알아보는 표식. 읽을 수 없으면 빈 문자열."""
    try:
        info = Path(path).stat()
    except OSError:
        return ""
    return f"{int(info.st_mtime_ns)}:{info.st_size}"


@dataclass
class PatchCache:
    """image_id → 격자 특징 (높이, 너비, 차원). 파일이 바뀌지 않은 것만 유효하다."""

    schema: str
    rows: dict[str, dict] = field(default_factory=dict)   # id → {"row": n, "stamp": s}
    shape: tuple[int, int, int] | None = None             # 격자 모양
    stored: np.ndarray | None = None                      # memmap (읽기 전용)
    fresh: dict[str, np.ndarray] = field(default_factory=dict)   # 이번에 새로 뽑은 것
    fresh_stamps: dict[str, str] = field(default_factory=dict)

    def get(self, image_id: str, path: str | os.PathLike[str]) -> np.ndarray | None:
        """캐시된 격자. 없거나 파일이 바뀌었으면 None. 항상 float32로 돌려준다."""
        image_id = str(image_id)
        current = stamp(path)
        if not current:
            return None      # 파일을 읽을 수 없으면 캐시를 믿지 않는다 — 지워졌을 수 있다

        grid = self.fresh.get(image_id)
        if grid is not None:
            return grid

        entry = self.rows.get(image_id)
        if entry is None or entry.get("stamp") != current or self.stored is None:
            return None
        index = int(entry["row"])
        if not 0 <= index < len(self.stored):
            return None
        return np.asarray(self.stored[index], dtype=np.float32)

    def put(self, image_id: str, path: str | os.PathLike[str], grid: np.ndarray) -> None:
        current = stamp(path)
        if not current:
            return
        grid = np.asarray(grid, dtype=np.float32)
        if self.shape is None:
            self.shape = tuple(int(v) for v in grid.shape)
        elif tuple(grid.shape) != self.shape:
            return       # 격자 모양이 섞이면 한 배열에 담을 수 없다
        self.fresh[str(image_id)] = grid
        # 표식은 **지금** 재 둔다. 저장할 때 다시 재면 그 사이에 바뀐 파일을 놓친다.
        self.fresh_stamps[str(image_id)] = current

    @property
    def added(self) -> int:
        return len(self.fresh)

    def __len__(self) -> int:
        return len(set(self.rows) | set(self.fresh))


def load(anomaly_config) -> PatchCache:
    """캐시를 연다. 배열은 memmap이라 **여는 순간에는 읽지 않는다.**"""
    schema = schema_fingerprint(anomaly_config)
    folder = cache_dir(schema)
    index_path, data_path = folder / INDEX_NAME, folder / DATA_NAME
    if not (index_path.exists() and data_path.exists()):
        return PatchCache(schema=schema)

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        rows = {str(k): v for k, v in payload["rows"].items()}
        shape = tuple(int(v) for v in payload["shape"])
        stored = np.load(data_path, mmap_mode="r")
    except (OSError, ValueError, KeyError, TypeError):
        # 저장 도중 죽어 잘린 파일 등. 캐시는 다시 만들면 되므로 조용히 버린다.
        return PatchCache(schema=schema)

    if stored.ndim != 4 or tuple(stored.shape[1:]) != shape:
        return PatchCache(schema=schema)
    return PatchCache(schema=schema, rows=rows, shape=shape, stored=stored)


def save(cache: PatchCache) -> None:
    """새로 뽑은 것을 합쳐 저장한다. 실패해도 예외를 올리지 않는다.

    **새로 뽑은 것이 없으면 아무것도 하지 않는다** — 두 번째 실행부터는 읽기만 한다.

    쓸 때도 한 줄씩 옮긴다. 0.12GB를 통째로 메모리에 올렸다가 쓰면 그 순간만 메모리가
    두 배가 되는데, 옮기는 일에 그럴 이유가 없다.
    """
    if not cache.fresh or cache.shape is None:
        return

    keep = [
        (image_id, entry) for image_id, entry in cache.rows.items()
        if image_id not in cache.fresh and cache.stored is not None
        and 0 <= int(entry.get("row", -1)) < len(cache.stored)
    ]
    total = len(keep) + len(cache.fresh)
    folder = cache_dir(cache.schema)

    try:
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f"{DATA_NAME}.{os.getpid()}.tmp.npy"
        target = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=STORED_DTYPE, shape=(total, *cache.shape)
        )
        index: dict[str, dict] = {}
        position = 0
        for image_id, entry in keep:
            target[position] = cache.stored[int(entry["row"])]
            index[image_id] = {"row": position, "stamp": entry.get("stamp", "")}
            position += 1
        for image_id, grid in cache.fresh.items():
            target[position] = grid.astype(STORED_DTYPE)
            index[image_id] = {"row": position, "stamp": cache.fresh_stamps.get(image_id, "")}
            position += 1
        target.flush()
        del target

        os.replace(temporary, folder / DATA_NAME)
        (folder / INDEX_NAME).write_text(
            json.dumps({"shape": list(cache.shape), "rows": index}), encoding="utf-8"
        )
    except (OSError, ValueError):
        return


def clear(anomaly_config=None) -> None:
    """캐시를 지운다. 설정을 주면 그 설정 것만, 안 주면 전부."""
    import shutil

    root = config.cache_dir() / PATCHES_DIR
    target = cache_dir(schema_fingerprint(anomaly_config)) if anomaly_config else root
    try:
        shutil.rmtree(target, ignore_errors=True)
    except OSError:
        pass


def summary() -> dict:
    """화면에 띄울 현황. 설정별 폴더를 모두 합친 값."""
    root = config.cache_dir() / PATCHES_DIR
    if not root.is_dir():
        return {"count": 0, "size_mb": 0.0, "variants": 0}

    count = 0
    size = 0
    variants = 0
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        variants += 1
        for item in folder.iterdir():
            try:
                size += item.stat().st_size
            except OSError:
                continue
        index_path = folder / INDEX_NAME
        try:
            count += len(json.loads(index_path.read_text(encoding="utf-8"))["rows"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {"count": count, "size_mb": size / 1024**2, "variants": variants}


def grids_for(
    rows,
    model,
    *,
    cache: PatchCache | None = None,
    progress=None,
):
    """행 목록에 대한 격자 특징을 순서대로 내놓는다 (캐시 우선).

    캐시에 있으면 **이미지 파일을 열지도 않는다** — 읽기 12.3ms가 통째로 빠진다.
    없으면 뽑아서 캐시에 담는다. 저장은 부르는 쪽이 `save()`로 한다(학습이 실패해도
    뽑아 둔 특징은 남기고 싶을 때가 있어 저장 시점을 밖에 둔다).

    `rows`는 `path_abs`와 `image_id`를 가진 DataFrame이고, 읽을 수 없는 이미지는 건너뛴다.
    """
    from . import viz

    cache = load(model.config) if cache is None else cache
    total = len(rows)
    for order, (_, row) in enumerate(rows.iterrows(), start=1):
        path = row["path_abs"]
        image_id = str(row["image_id"])

        grid = cache.get(image_id, path)
        if grid is None:
            image = viz.load_rgb(path)
            if image is None:
                if progress is not None:
                    progress(order, total)
                continue
            grid = model.grid_features(image)
            cache.put(image_id, path, grid)

        if progress is not None:
            progress(order, total)
        yield image_id, grid
