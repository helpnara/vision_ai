"""2단계 라벨링 API 배선.

코어(`labeling` `boxes` `detection`)는 각자의 테스트가 본다. 여기서는 라우터가 코어를 바르게
부르고 응답이 JSON으로 나가며, 화면이 기대하는 흐름(큐 조회 → 저장 → 큐에서 빠짐)이 도는지 본다.
"""

from __future__ import annotations

import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from vision_ai import boxes as box_store
from vision_ai import config, ingest, labeling, storage


@pytest.fixture
def client(sandbox):
    from server import state
    from server.main import app

    state.clear()
    return TestClient(app)


@pytest.fixture
def seeded(client):
    """마스크가 있는 합성 데이터 (mvtec 배치는 ground_truth 마스크를 함께 만든다)."""
    out = ingest.generate_synthetic(categories=["metal"], n_normal=8, n_defect=6, size=96, seed=3, layout="mvtec")
    ingest.ingest_folder(out, source=ingest.SYNTHETIC_SOURCE, layout="mvtec")
    return client


@pytest.fixture
def seeded_visa(client):
    """분할이 비어 있는 합성 데이터 — visa 배치는 폴더에 train/test가 없다."""
    out = ingest.generate_synthetic(categories=["metal"], n_normal=8, n_defect=6, size=96, seed=3, layout="visa")
    ingest.ingest_folder(out, source=ingest.SYNTHETIC_SOURCE, layout="visa")
    return client


def wait_job(client: TestClient, job_id: str, timeout: int = 120) -> dict:
    for _ in range(timeout * 10):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("잡이 끝나지 않았다")


# --- 빈 프로젝트 ---------------------------------------------------------------

def test_empty_project_is_json_safe_on_every_tab(client):
    assert client.get("/api/labeling/overview").json()["total"] == 0
    assert client.get("/api/labeling/queue").json() == {"total_images": 0, "categories": ["(전체)"], "ids": []}
    assert client.get("/api/labeling/video-frames").json() == {"videos": []}
    assert client.get("/api/labeling/verify").json()["total_images"] == 0
    assert client.get("/api/labeling/mapping").json()["unmapped"] == []
    assert client.get("/api/labeling/split").json() == {"total_images": 0}
    assert client.get("/api/labeling/status").json()["stats"]["total"] == 0
    assert client.get("/api/labeling/boxes").json()["summary"]["boxes"] == 0


# --- 라벨 검수 큐 ---------------------------------------------------------------

def test_queue_item_save_and_drain(seeded):
    client = seeded
    queue = client.get("/api/labeling/queue?mode=unverified").json()
    assert queue["total_images"] > 0
    assert "(전체)" in queue["categories"] and "metal" in queue["categories"]
    before = len(queue["ids"])
    assert before > 0

    image_id = queue["ids"][0]
    item = client.get(f"/api/labeling/item/{image_id}").json()
    assert item["image_id"] == image_id
    assert item["width"] == 96 and item["height"] == 96
    assert item["label_source"] == "folder" and item["verified"] is False
    assert item["boxes"] == []

    saved = client.post("/api/labeling/save", json={
        "image_id": image_id, "label": "normal", "defect_type": None, "boxes": [], "note": "ok",
    }).json()
    assert saved["ok"] is True

    after = client.get("/api/labeling/queue?mode=unverified").json()["ids"]
    assert image_id not in after
    assert len(after) == before - 1

    row = labeling.resolve().set_index("image_id").loc[image_id]
    assert row["label_source"] == "human" and bool(row["verified"]) is True


def test_boxes_round_trip_and_preview_image(seeded):
    client = seeded
    resolved = labeling.resolve()
    image_id = str(resolved[resolved["label"] == config.LABEL_DEFECT].iloc[0]["image_id"])

    item = client.get(f"/api/labeling/item/{image_id}").json()
    assert item["auto_roi"] is not None, "mvtec 배치에는 마스크가 있어야 한다"

    assert client.post("/api/labeling/save", json={
        "image_id": image_id, "label": "defect", "defect_type": None, "boxes": [],
    }).status_code == 400

    boxes = [{"x": 5, "y": 6, "w": 20, "h": 10, "label": "scratch"},
             {"x": 40, "y": 40, "w": 12, "h": 12, "label": "unspecified"}]
    client.post("/api/labeling/save", json={
        "image_id": image_id, "label": "defect", "defect_type": "scratch", "boxes": boxes,
    })
    again = client.get(f"/api/labeling/item/{image_id}").json()
    assert [(b["x"], b["y"], b["w"], b["h"]) for b in again["boxes"]] == [(5, 6, 20, 10), (40, 40, 12, 12)]
    assert again["defect_type"] == "scratch"
    row = labeling.resolve().set_index("image_id").loc[image_id]
    assert (int(row["roi_x"]), int(row["roi_w"])) == (5, 20)   # 첫 박스가 대표 ROI

    shot = client.get(f"/api/labeling/preview/{image_id}?boxes=5,6,20,10,scratch;40,40,12,12")
    assert shot.status_code == 200 and shot.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/labeling/preview/{image_id}?boxes=garbage").status_code == 400
    assert client.get("/api/labeling/preview/nope").status_code == 404

    # 정상으로 되돌리면 박스도 비운다
    client.post("/api/labeling/save", json={"image_id": image_id, "label": "normal"})
    assert client.get(f"/api/labeling/item/{image_id}").json()["boxes"] == []


# --- 폴더 라벨 검증 ---------------------------------------------------------------

def test_verify_sample_then_confirm(seeded):
    client = seeded
    sample = client.get("/api/labeling/verify?size=4&seed=1").json()
    assert sample["pending"] > 0 and len(sample["items"]) == 4
    first = sample["items"][0]
    flipped = "defect" if first["label"] == "normal" else "normal"
    decisions = [{"image_id": first["image_id"], "label": flipped}] + [
        {"image_id": it["image_id"], "label": it["label"]} for it in sample["items"][1:]
    ]
    out = client.post("/api/labeling/verify", json={"decisions": decisions}).json()
    assert out == {"count": 4, "changed": 1}
    assert client.get("/api/labeling/verify?size=4&seed=1").json()["pending"] == sample["pending"] - 4


# --- 결함 유형 정규화 -------------------------------------------------------------

def test_mapping_is_saved_and_applied(seeded):
    client = seeded
    body = client.get("/api/labeling/mapping").json()
    assert body["mapping"]["total"] > 0
    assert client.post("/api/labeling/mapping", json={"choices": {"weird": "nothing"}}).status_code == 400
    assert client.post("/api/labeling/mapping", json={"choices": {"weird": "stain"}}).json() == {"count": 1}
    assert labeling.load_type_map()["weird"] == "stain"
    rows = client.get("/api/labeling/mapping").json()["mapping"]["rows"]
    assert any(r["원본 유형"] == "weird" and r["표준 유형(한글)"] == "얼룩 / 오염" for r in rows)


# --- 데이터 분할 ------------------------------------------------------------------

def test_split_run_state_and_clear(seeded_visa):
    client = seeded_visa
    state = client.get("/api/labeling/split").json()
    assert state["has_split"] is False and state["crosstab"] is None
    assert state["gap"]["complete"] is False
    assert [m["key"] for m in state["split_modes"]] == list(labeling.SPLIT_MODES)

    assert client.post("/api/labeling/split/run", json={"train": 0.6, "val": 0.4, "test": 0.0}).status_code == 400

    out = client.post("/api/labeling/split/run", json={
        "train": 0.6, "val": 0.2, "test": 0.2, "seed": 1, "labeled_only": True, "mode": "group",
    }).json()
    assert out["count"] == len(storage.load_manifest())

    state = client.get("/api/labeling/split").json()
    assert state["has_split"] is True and state["gap"]["complete"] is True
    assert state["crosstab"]["columns"][:2] == ["category", "label"]
    assert client.get("/api/labeling/overview").json()["split_done"] is True

    client.post("/api/labeling/split/clear")
    assert client.get("/api/labeling/split").json()["has_split"] is False


def test_split_csv_import_matches_by_path_suffix(seeded):
    client = seeded
    manifest = storage.load_manifest()
    lines = ["image,split"] + [f"{p},train" for p in manifest["path"].astype(str).head(3)] + ["ghost.png,test"]
    csv = "\n".join(lines).encode("utf-8")
    out = client.post("/api/labeling/split/import", files={"file": ("split.csv", csv, "text/csv")}).json()
    assert out == {"count": 3, "unmatched": 1}

    bad = client.post("/api/labeling/split/import", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 400 and "임포트 실패" in bad.json()["detail"]


# --- 라벨 현황 · 내려받기 ----------------------------------------------------------

def test_status_and_text_downloads(seeded):
    client = seeded
    assert client.get("/api/labeling/export/events").status_code == 404   # 아직 이벤트 없음
    image_id = str(storage.load_manifest().iloc[0]["image_id"])
    client.post("/api/labeling/save", json={"image_id": image_id, "label": "normal"})

    body = client.get("/api/labeling/status").json()
    assert body["stats"]["human"] == 1
    assert body["events_total"] == 1 and body["events"]["rows"][0]["image_id"] == image_id
    assert any(c["name"] == "human" for c in body["label_sources"])

    resolved = client.get("/api/labeling/export/resolved").json()
    assert resolved["filename"] == "resolved_labels.csv" and "image_id" in resolved["text"]
    assert client.get("/api/labeling/export/events").json()["filename"] == "labels.csv"
    assert client.get("/api/labeling/export/nothing").status_code == 404


# --- 결함 박스 -----------------------------------------------------------------------

def test_mask_boxes_job_then_exports(seeded):
    client = seeded
    before = client.get("/api/labeling/boxes").json()
    assert before["summary"]["boxes"] == 0 and before["mask"]["ready"] > 0

    job_id = client.post("/api/labeling/boxes/from-masks").json()["job_id"]
    done = wait_job(client, job_id)
    assert done["status"] == "done", done["error"]
    assert done["result"]["images"] == before["mask"]["ready"]
    assert done["result"]["boxes"] >= done["result"]["images"]

    after = client.get("/api/labeling/boxes").json()
    assert after["summary"]["boxes"] == done["result"]["boxes"]
    assert after["mask"]["ready"] == 0 and after["mask"]["already"] == before["mask"]["ready"]
    assert after["class_names"]
    assert after["readiness"]["ready"] is False      # 50장 미만

    assert "box_id" in client.get("/api/labeling/export/boxes").json()["text"]
    coco = client.get("/api/labeling/export/coco").json()
    assert coco["filename"] == "annotations_coco.json" and '"annotations"' in coco["text"]

    zipped = client.get("/api/labeling/export-yolo.zip")
    assert zipped.status_code == 200 and zipped.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(zipped.content)).namelist()
    assert "classes.txt" in names and any(n.startswith("labels/") for n in names)


def test_adopting_legacy_rois_moves_them_into_boxes(seeded):
    client = seeded
    image_id = str(storage.load_manifest().iloc[0]["image_id"])
    labeling.record_label(image_id, label="defect", defect_type="dent", roi=(1, 2, 10, 10))
    assert client.get("/api/labeling/boxes").json()["adoptable"] == 1
    assert client.post("/api/labeling/boxes/adopt").json() == {"moved": 1}
    assert len(box_store.for_image(image_id)) == 1


def test_detection_export_reports_why_training_must_not_start(seeded):
    client = seeded
    # 박스가 하나도 없으면 결함 이미지는 전부 빠지고 «결함이 한 장도 없다»가 막는 이유가 된다
    out = client.post("/api/labeling/detection/export", json={"copy_images": False}).json()
    assert "결함 이미지가 한 장도 없습니다" in out["blocking"]
    assert out["command"].startswith("pip install ultralytics")
    assert any("라벨이 없어" in note for note in out["notes"])
    assert out["root"].endswith("detection")
