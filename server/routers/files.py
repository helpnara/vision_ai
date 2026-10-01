"""이미지·영상·내려받기 파일 서빙.

Streamlit은 `st.image(array)`로 배열을 바로 그렸지만 브라우저 화면은 URL이 필요하다.
manifest의 `image_id`로 원본을 찾아 주고, 큰 이미지는 폭을 줄여 보낸다 — 1404px PCB
이미지를 갤러리 칸 한 칸에 원본으로 보내면 화면 하나에 수십 MB가 된다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, Response

from vision_ai import config, storage, video, viz

from ..common import inside, not_found

router = APIRouter(prefix="/files", tags=["files"])

THUMB_WIDTH = 480
"""갤러리 칸에 보내는 폭. 4열 갤러리 한 칸이 이보다 넓어지는 일은 거의 없다."""

MAX_WIDTH = 1600
"""«원본»을 요청해도 이 폭을 넘기지 않는다. 박스를 그릴 때도 이 폭이면 충분하다."""


def _encode_jpeg(rgb: np.ndarray, *, width: int | None, quality: int = 88) -> bytes:
    array = np.asarray(rgb)
    if width and array.shape[1] > width:
        scale = width / array.shape[1]
        array = cv2.resize(array, (width, max(1, round(array.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(array, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise not_found("이미지를 변환하지 못했습니다")
    return buffer.tobytes()


def image_response(rgb: np.ndarray, *, width: int | None = THUMB_WIDTH) -> Response:
    return Response(content=_encode_jpeg(rgb, width=width), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


def _manifest_row(image_id: str):
    df = storage.load_manifest()
    if df.empty:
        return None
    rows = df[df["image_id"].astype(str) == str(image_id)]
    return None if rows.empty else rows.iloc[0]


@router.get("/image/{image_id}")
def image(image_id: str, w: int = Query(THUMB_WIDTH, ge=0, le=MAX_WIDTH)) -> Response:
    """manifest에 등록된 이미지. `w=0`이면 상한(MAX_WIDTH) 안에서 원본 크기."""
    row = _manifest_row(image_id)
    if row is None:
        raise not_found(f"이미지가 없습니다: {image_id}")
    rgb = viz.load_rgb(storage.resolve_path(row["path"]))
    if rgb is None:
        raise not_found(f"이미지를 읽을 수 없습니다: {row['path']}")
    return image_response(rgb, width=w or MAX_WIDTH)


@router.get("/image/{image_id}/size")
def image_size(image_id: str) -> dict:
    row = _manifest_row(image_id)
    if row is None:
        raise not_found(f"이미지가 없습니다: {image_id}")
    rgb = viz.load_rgb(storage.resolve_path(row["path"]))
    if rgb is None:
        raise not_found("이미지를 읽을 수 없습니다")
    return {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])}


@router.get("/path")
def file_at(path: str, w: int = Query(THUMB_WIDTH, ge=0, le=MAX_WIDTH)) -> Response:
    """프로젝트 데이터·산출물 폴더 안의 파일 (추출 표본, 판정 정지 화면 등).

    아무 경로나 열어 주지 않는다 — 프로젝트 자료 폴더 안의 파일만 돌려준다.
    """
    target = Path(path).expanduser().resolve()
    if not inside(target, (config.DATA_HOME, config.ARTIFACT_HOME)):
        raise not_found("프로젝트 폴더 밖의 파일은 열 수 없습니다")
    if not target.is_file():
        raise not_found(f"파일이 없습니다: {target}")
    if target.suffix.lower() in config.IMAGE_EXTENSIONS:
        rgb = viz.load_rgb(target)
        if rgb is None:
            raise not_found("이미지를 읽을 수 없습니다")
        return image_response(rgb, width=w or MAX_WIDTH)
    return FileResponse(target)


@router.get("/video")
def video_file(path: str) -> FileResponse:
    """영상 파일 그대로 (브라우저 `<video>`용). 재생 가능 여부는 코덱에 달렸다 (playback.CODECS)."""
    target = Path(path).expanduser()
    # 판정본은 WebM(VP8)으로 만든다 (playback.CODECS — 브라우저가 mp4v를 못 틀기 때문).
    # video.VIDEO_EXTENSIONS 는 «입력으로 받는 영상»의 목록이라 .webm 이 없다. 여기서 더한다.
    playable = video.is_video(target) or target.suffix.lower() == ".webm"
    if not target.is_file() or not playable:
        raise not_found(f"영상 파일이 없습니다: {target}")
    media = "video/webm" if target.suffix.lower() == ".webm" else "video/mp4"
    return FileResponse(target, media_type=media)
