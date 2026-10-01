"""1단계 수집 API 배선 테스트.

코어(ingest·video·framing)는 각자의 테스트가 본다. 여기서는 라우터가 코어를 바르게 부르고,
응답이 JSON으로 나가며(NaN 없음), 잡이 끝까지 돌아 결과를 돌려주는지만 본다.
"""

from __future__ import annotations

import io
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vision_ai import ingest, storage


@pytest.fixture
def client(sandbox):
    from server import state
    from server.main import app

    state.clear()
    return TestClient(app)


def wait_job(client: TestClient, job_id: str, timeout: int = 120) -> dict:
    for _ in range(timeout * 10):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("잡이 끝나지 않았다")


def _png_bytes(size: int = 64, shade: int = 120) -> bytes:
    image = np.full((size, size, 3), shade, np.uint8)
    cv2.line(image, (5, 5), (size - 5, size - 5), (30, 30, 30), 2)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


# --- 선택지·카탈로그 ---------------------------------------------------------------

def test_options_marks_the_empty_project_and_lists_choices(client):
    body = client.get("/api/ingest/options").json()
    assert body["has_data"] is False
    assert body["default_dataset"]["key"] == "visa"
    assert "visa" in body["layouts"] and "custom" in body["layouts"]
    assert body["surface_styles"][:2] == ["pcb_green", "pcb_blue"]
    assert body["model_inputs"][0]["px"] == 640


def test_catalog_puts_the_default_dataset_first_with_a_star(client):
    body = client.get("/api/ingest/catalog").json()
    assert body["table"]["rows"][0]["데이터셋"].startswith("⭐ ")
    assert body["datasets"][0]["is_default"] is True
    assert body["help"]["key"]


# --- 🧪 합성 샘플 → manifest ---------------------------------------------------------

def test_synthetic_job_registers_images_into_the_manifest(client):
    job_id = client.post("/api/ingest/synthetic", json={
        "categories": ["pcb_green"], "n_normal": 6, "n_defect": 4, "size": 64, "seed": 1,
        "layout": "visa", "overwrite": True,
    }).json()["job_id"]
    body = wait_job(client, job_id)
    assert body["status"] == "done", body["error"]
    assert body["result"]["added"] == 10
    assert "신규 등록 10건" in body["result"]["message"]
    assert len(storage.load_manifest()) == 10
    assert client.get("/api/ingest/options").json()["has_data"] is True


def test_synthetic_without_categories_is_rejected(client):
    assert client.post("/api/ingest/synthetic", json={"categories": []}).status_code == 400


# --- 📁 폴더 임포트 -----------------------------------------------------------------

def test_folder_preview_counts_images_and_parses_the_layout(client):
    out = ingest.generate_synthetic(categories=["fabric"], n_normal=3, n_defect=4, size=64, seed=2, layout="mvtec")
    body = client.get("/api/ingest/folder/preview", params={"root": str(out), "layout": "mvtec"}).json()
    # 마스크도 이미지 파일이라 함께 센다 (등록 때 제외된다)
    assert body["total"] == 3 + 4 + 4
    assert body["preview"]["rows"]
    assert {"상대경로", "카테고리", "분할", "라벨", "결함유형", "마스크"} <= set(body["preview"]["columns"])
    assert any(r["카테고리"] == "fabric" for r in body["preview"]["rows"])


def test_folder_preview_of_a_missing_folder_is_404_with_the_screen_sentence(client, tmp_path):
    res = client.get("/api/ingest/folder/preview", params={"root": str(tmp_path / "nope")})
    assert res.status_code == 404
    assert "폴더를 찾을 수 없습니다" in res.json()["detail"]


def test_folder_ingest_job_skips_masks_and_reports_progress(client):
    out = ingest.generate_synthetic(categories=["fabric"], n_normal=3, n_defect=4, size=64, seed=2, layout="mvtec")
    job_id = client.post("/api/ingest/folder/ingest", json={
        "root": str(out), "layout": "mvtec", "source": "demo", "include_masks": False,
    }).json()["job_id"]
    body = wait_job(client, job_id)
    assert body["status"] == "done", body["error"]
    assert body["result"]["added"] == 7
    assert body["result"]["skipped_masks"] == 4
    assert body["fraction"] == 1.0
    assert set(storage.load_manifest()["source"]) == {"demo"}


# --- ⬆️ 업로드 --------------------------------------------------------------------

def test_upload_registers_images_and_names_unreadable_files(client):
    files = [
        ("files", ("a.png", io.BytesIO(_png_bytes(shade=100)), "image/png")),
        ("files", ("b.png", io.BytesIO(_png_bytes(shade=160)), "image/png")),
        ("files", ("junk.png", io.BytesIO(b"not an image"), "image/png")),
    ]
    res = client.post("/api/ingest/upload", files=files,
                      data={"category": "mug", "label": "defect", "defect_type": "scratch"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["added"] == 2
    assert body["failed"] == ["junk.png"]
    df = storage.load_manifest()
    assert set(df["category"]) == {"mug"}
    assert set(df["defect_type"]) == {"scratch"}


def test_upload_drops_the_defect_type_when_the_label_is_not_defect(client):
    files = [("files", ("a.png", io.BytesIO(_png_bytes()), "image/png"))]
    client.post("/api/ingest/upload", files=files, data={"category": "", "label": "normal", "defect_type": "scratch"})
    df = storage.load_manifest()
    assert df.iloc[0]["defect_type"] == "none"
    assert df.iloc[0]["category"] == "uncategorized"


# --- 🎞️ 영상: 선택기 → probe → 계획 → 추출 -------------------------------------------

def test_video_sample_is_listed_probed_and_extracted_as_a_job(client):
    assert client.get("/api/ingest/videos").json()["videos"] == []

    made = client.post("/api/ingest/videos/sample").json()
    assert made["path"].endswith("sample.mp4")
    listed = client.get("/api/ingest/videos").json()
    assert [v["path"] for v in listed["videos"]] == [made["path"]]
    assert "MB" in listed["videos"][0]["label"] or "KB" in listed["videos"][0]["label"]

    assert client.get("/api/ingest/videos/check", params={"path": made["path"]}).status_code == 200

    info = client.get("/api/ingest/video/probe", params={"path": made["path"]}).json()
    assert info["fps"] == pytest.approx(30, abs=0.5)
    assert info["frame_count"] == 300
    assert (info["width"], info["height"]) == (640, 480)

    plan = client.post("/api/ingest/video/plan", json={"path": made["path"], "mode": "rate", "per_second": 2.0}).json()
    assert plan["stride"] == 15
    assert plan["expected_frames"] == 20
    assert plan["feasible"] is True

    job_id = client.post("/api/ingest/video/extract", json={
        "path": made["path"], "stride": plan["stride"], "category": "video", "dedupe": False, "check_quality": False,
    }).json()["job_id"]
    body = wait_job(client, job_id)
    assert body["status"] == "done", body["error"]
    result = body["result"]
    assert result["kept"] == 20
    assert result["added"] == 20
    assert result["mostly_dropped"] is False
    assert len(result["samples"]) == 4
    # 추출 표본은 프로젝트 폴더 안에 있으므로 pathUrl로 열린다
    assert client.get("/api/files/path", params={"path": result["samples"][0], "w": 64}).status_code == 200
    df = storage.load_manifest()
    assert set(df["group"]) == {result["video_id"]}


def test_video_plan_errors_from_the_core_become_400(client):
    made = client.post("/api/ingest/videos/sample").json()
    res = client.post("/api/ingest/video/plan", json={"path": made["path"], "mode": "rate", "per_second": 0})
    assert res.status_code == 400
    assert "0보다 커야" in res.json()["detail"]

    res = client.post("/api/ingest/video/plan", json={
        "path": made["path"], "mode": "process", "field_of_view_m": 0.1, "speed_mps": 10.0, "frames_per_object": 3,
    })
    assert res.status_code == 200
    assert res.json()["feasible"] is False
    assert "fps 이상" in res.json()["reason"]


def test_video_check_explains_a_missing_or_non_video_path(client, tmp_path):
    missing = client.get("/api/ingest/videos/check", params={"path": str(tmp_path / "ghost.mp4")})
    assert missing.status_code == 404
    assert "파일을 찾을 수 없습니다" in missing.json()["detail"]
    text = tmp_path / "note.txt"
    text.write_text("x")
    wrong = client.get("/api/ingest/videos/check", params={"path": str(text)})
    assert wrong.status_code == 400
    assert "영상 파일이 아닙니다" in wrong.json()["detail"]


def test_video_list_flags_a_scan_folder_that_does_not_exist(client, tmp_path):
    body = client.get("/api/ingest/videos", params={"scan": str(tmp_path / "nas")}).json()
    assert body["scan_dir_missing"] is True


def test_video_upload_lands_in_the_project_video_folder(client):
    made = client.post("/api/ingest/videos/sample").json()
    data = open(made["path"], "rb").read()
    res = client.post("/api/ingest/videos/upload", files={"file": ("line1.mp4", io.BytesIO(data), "video/mp4")})
    assert res.status_code == 200, res.text
    assert res.json()["name"] == "line1.mp4"
    names = {v["name"] for v in client.get("/api/ingest/videos").json()["videos"]}
    assert {"sample.mp4", "line1.mp4"} <= names
    refused = client.post("/api/ingest/videos/upload", files={"file": ("a.txt", io.BytesIO(b"x"), "text/plain")})
    assert refused.status_code == 400


def test_framing_check_returns_a_verdict_and_levers(client):
    body = client.post("/api/ingest/video/framing", json={
        "sensor_px": 640, "fov_m": 1.0, "defect_mm": 10.0, "model": "검출 (YOLO 640)", "tiles": 1,
    }).json()
    assert body["native_px"] == pytest.approx(6.4)
    assert body["verdict"] == "불가"
    assert body["levers"]
    assert body["hint"] is None
    assert body["smallest_mm"] > 10


# --- 📊 수집 현황 -----------------------------------------------------------------

def test_status_is_empty_before_any_ingest(client):
    assert client.get("/api/ingest/status").json() == {"empty": True}


def test_status_is_json_safe_and_offers_preview_manifest_and_source_removal(client):
    out = ingest.generate_synthetic(categories=["pcb_green", "fabric"], n_normal=8, n_defect=4, size=64, seed=3)
    ingest.ingest_folder(out, source=ingest.SYNTHETIC_SOURCE, layout="visa")

    res = client.get("/api/ingest/status")
    assert res.status_code == 200
    body = res.json()
    assert body["empty"] is False
    assert body["stats"]["total"] == 24
    assert {c["name"] for c in body["by_category"]} == {"pcb_green", "fabric"}
    assert body["crosstab"]["columns"][0] == "source"
    assert {"정상", "결함"} <= set(body["crosstab"]["columns"])
    assert body["quality"]["mean_blur"] is None or isinstance(body["quality"]["mean_blur"], float)
    assert body["sources"] == ["synthetic"]
    assert "NaN" not in res.text

    preview = client.get("/api/ingest/preview", params={"category": "fabric", "label": "defect", "count": 4}).json()
    assert len(preview["items"]) == 4
    assert all(item["exists"] for item in preview["items"])
    assert all("(unspecified)" in item["caption"] for item in preview["items"])
    assert client.get("/api/ingest/preview", params={"category": "ghost"}).json()["items"] == []

    manifest = client.get("/api/ingest/manifest").json()
    assert manifest["table"]["total"] == 24
    assert manifest["csv"].startswith("image_id,path,source")

    removed = client.delete("/api/ingest/sources", params={"name": "synthetic"}).json()
    assert removed["removed"] == 24
    assert client.get("/api/ingest/status").json() == {"empty": True}
