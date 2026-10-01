"""웹 API 뼈대 — 프로젝트·홈·설정·잡·파일 서빙.

코어 로직은 각자의 테스트가 본다. 여기서는 **배선**만 본다: 라우터가 코어를 바르게 부르고,
응답이 JSON으로 나가며(NaN 없음), 잡이 끝까지 돌아 결과를 돌려주는가.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from vision_ai import config, ingest, storage


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


def test_health_names_the_active_project(client):
    body = client.get("/api/health").json()
    assert body == {"ok": True, "project": config.active_project()}


def test_overview_is_json_safe_on_an_empty_project(client):
    body = client.get("/api/home/overview").json()
    assert body["stats"]["total"] == 0
    assert body["label_stats"] is None
    assert body["quickstart_available"] is True
    assert body["next_step"]["route"] == "/ingest"
    assert body["validation"]["available"] in (True, False)


def test_overview_counts_registered_images(client):
    out = ingest.generate_synthetic(categories=["metal"], n_normal=6, n_defect=4, size=64, seed=1)
    ingest.ingest_folder(out, source=ingest.SYNTHETIC_SOURCE, layout="mvtec")
    body = client.get("/api/home/overview").json()
    assert body["stats"]["total"] == len(storage.load_manifest()) > 0
    assert body["label_stats"]["split_assigned"] == 0
    assert body["quickstart_available"] is False


def test_quickstart_runs_as_a_job_and_promotes_a_model(client):
    job_id = client.post("/api/home/quickstart").json()["job_id"]
    body = wait_job(client, job_id)
    assert body["status"] == "done", body["error"]
    assert body["result"]["ok"] is True
    assert body["result"]["version"] == "v001"
    assert body["fraction"] == 1.0
    assert not storage.load_manifest().empty


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_projects_can_be_created_switched_and_removed(client):
    first = client.get("/api/projects").json()
    assert first["active"] == "default"

    made = client.post("/api/projects", json={"name": "2공장 도장라인"}).json()
    slug = made["created"]["slug"]
    assert made["active"] == slug
    assert config.active_project() == slug

    client.post("/api/projects/use", json={"slug": "default"})
    assert config.active_project() == "default"

    client.post(f"/api/projects/{slug}/rename", json={"name": "새 이름"})
    names = {p["slug"]: p["name"] for p in client.get("/api/projects").json()["projects"]}
    assert names[slug] == "새 이름"

    after = client.delete(f"/api/projects/{slug}").json()
    assert slug not in {p["slug"] for p in after["projects"]}


def test_switching_to_an_unknown_project_is_rejected(client):
    assert client.post("/api/projects/use", json={"slug": "ghost"}).status_code == 400


def test_settings_round_trip_and_reset(client):
    body = client.get("/api/settings").json()
    assert body["values"]["target_recall"] == pytest.approx(0.95)
    assert body["changed"] == {}

    values = dict(body["values"], target_recall=0.9, volume=2000)
    saved = client.put("/api/settings", json=values).json()
    assert saved["clamped"] is False
    assert saved["values"]["volume"] == 2000
    assert set(saved["changed"]) == {"target_recall", "volume"}
    assert saved["preview"]["passes"] is True   # 실측 0.925 ≥ 0.9

    reset = client.post("/api/settings/reset").json()
    assert reset["changed"] == {}


def test_settings_preview_reports_why_the_measured_model_fails(client):
    body = client.get("/api/settings").json()
    values = dict(body["values"], target_recall=0.99)
    preview = client.post("/api/settings/preview", json=values).json()
    assert preview["passes"] is False
    assert any("재현율" in line for line in preview["missing"])


def test_image_is_served_by_id_and_shrunk(client):
    out = ingest.generate_synthetic(categories=["metal"], n_normal=2, n_defect=0, size=640, seed=2)
    ingest.ingest_folder(out, source=ingest.SYNTHETIC_SOURCE, layout="mvtec")
    image_id = str(storage.load_manifest().iloc[0]["image_id"])

    size = client.get(f"/api/files/image/{image_id}/size").json()
    assert size == {"width": 640, "height": 640}

    thumb = client.get(f"/api/files/image/{image_id}?w=120")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
    assert len(thumb.content) < len(client.get(f"/api/files/image/{image_id}?w=0").content)

    assert client.get("/api/files/image/missing").status_code == 404


def test_files_outside_the_project_folders_are_refused(client, tmp_path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("x")
    assert client.get(f"/api/files/path?path={outside}").status_code == 404


def test_spa_fallback_serves_index_for_client_routes(client):
    from server.main import WEB_DIST

    if not WEB_DIST.is_dir():
        pytest.skip("web/dist가 없습니다 (npm run build 전)")
    assert client.get("/labeling").status_code == 200
    assert "<div id=\"root\">" in client.get("/labeling").text
