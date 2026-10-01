"""4단계 운영관리 API 배선 테스트.

코어 로직은 각자의 테스트가 본다. 여기서는 라우터가 코어를 바르게 부르고, 응답이 JSON으로
나가며(NaN 없음), 잡이 끝까지 돌아 결과를 돌려주는가만 본다.

데모 데이터는 `quickstart.run()`으로 한 번만 만든다(약 15초, v001이 승격된다). 테스트는
**순서대로** 상태를 쌓는다 — 배치 추론이 로그를 만들고, 그 로그로 드리프트·성능 추이가
채워지고, 시나리오가 검수 정답을 남긴다. 매 테스트마다 데모를 다시 만들면 2분이 넘는다.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vision_ai import config, monitoring, quickstart, video


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """모듈 범위의 sandbox + 데모 한 바퀴."""
    root = tmp_path_factory.mktemp("ops")
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "DATA_HOME", root / "data")
    patch.setattr(config, "ARTIFACT_HOME", root / "artifacts")
    patch.setattr(config, "_active_project", config.DEFAULT_PROJECT)
    config.ensure_dirs()
    result = quickstart.run()
    assert result.ok, result.warnings
    yield result
    patch.undo()


@pytest.fixture(scope="module")
def client(demo):
    from server import state
    from server.main import app

    state.clear()
    return TestClient(app)


def wait_job(client: TestClient, job_id: str, timeout: int = 300) -> dict:
    for _ in range(timeout * 10):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("잡이 끝나지 않았다")


def _no_nan(value) -> None:
    """응답 어디에도 NaN이 없어야 한다 — JSON 파서가 거부한다."""
    if isinstance(value, float):
        assert not math.isnan(value)
    elif isinstance(value, dict):
        for item in value.values():
            _no_nan(item)
    elif isinstance(value, list):
        for item in value:
            _no_nan(item)


def test_registry_lists_the_promoted_demo_version(client):
    body = client.get("/api/operations/registry").json()
    assert body["versions"] == ["v001"]
    assert body["table"]["rows"][0]["상태"] == "서비스 중"
    assert body["options"] == []                       # 데모 실행은 이미 등록됐다
    assert body["rollback_target"] is None             # 한 번도 교체하지 않았다

    gate = client.get("/api/operations/registry/gate", params={"version": "v001"}).json()
    assert set(gate) == {"passed", "problems", "notes"}

    detail = client.get("/api/operations/registry/detail", params={"version": "v001"}).json()
    assert detail["run"]["kind"] == "anomaly"
    assert detail["baseline_samples"] > 0


def test_batch_inference_runs_as_a_job_and_fills_the_log(client):
    view = client.get("/api/operations/inference").json()
    assert view["usable"] == ["v001"]
    assert view["default_version"] == "v001"
    assert view["log"]["summary"]["total"] == 0

    target = client.get(
        "/api/operations/inference/target",
        params={"version": "v001", "scope": "전체", "limit": 60},
    ).json()
    assert target["count"] == 60

    job_id = client.post(
        "/api/operations/inference/run",
        json={"version": "v001", "scope": "전체", "limit": 60},
    ).json()["job_id"]
    body = wait_job(client, job_id)
    assert body["status"] == "done", body["error"]
    assert body["result"]["last"]["n"] == 60
    assert body["result"]["last"]["head"]["total"] == 50

    after = client.get("/api/operations/inference").json()
    assert after["log"]["summary"]["total"] == 60
    assert after["last"]["version"] == "v001"
    _no_nan(after)
    assert client.get("/api/operations/inference/log/csv").json()["csv"].startswith("logged_at")


def test_drift_answers_from_the_last_batch(client):
    body = client.get("/api/operations/drift").json()
    assert body["production"]["version"] == "v001"
    assert body["baseline"]["n_samples"] > 0
    assert body["has_batch"] is True
    # 60건은 최소 표본(200)에 못 미치므로 판정을 보류한다 — 0으로 적지 않는다
    assert body["summary"]["level"] == monitoring.LEVEL_INSUFFICIENT
    assert "무엇을 재는가" in body["table"]["columns"]
    _no_nan(body)


def test_performance_without_verified_labels_says_so(client):
    body = client.get("/api/operations/performance").json()
    assert body["log_empty"] is False
    assert body["feedback_empty"] is True              # 폴더 라벨은 정답으로 치지 않는다


def test_scenario_replays_three_months_and_fills_the_other_tabs(client):
    view = client.get("/api/operations/scenario").json()
    assert len(view["timeline"]) == 3
    assert view["result"] is None

    job_id = client.post("/api/operations/scenario/run").json()["job_id"]
    body = wait_job(client, job_id)
    assert body["status"] == "done", body["error"]
    result = body["result"]
    assert result["total_logged"] > 0
    assert result["table"]["total"] == 3
    assert result["days"] == 90
    _no_nan(result)

    again = client.get("/api/operations/scenario").json()
    assert again["result"]["total_logged"] == result["total_logged"]

    perf = client.get("/api/operations/performance").json()
    assert perf["feedback_empty"] is False
    assert perf["metrics"]["n"] > 0
    assert perf["by_version"]["rows"][0]["version"] == "v001"
    assert "trend" in perf
    _no_nan(perf)


def test_retraining_recomputes_when_inputs_change(client):
    body = client.get("/api/operations/retraining").json()
    assert body["defaults"]["new_label_threshold"] == 50
    keys = {s["key"] for s in body["decision"]["signals"]}
    assert "production" in keys and "log" in keys

    again = client.post(
        "/api/operations/retraining", json={"new_label_threshold": 1, "recall_margin": 0.3}
    ).json()
    assert any(s["key"] == "new_labels" for s in again["signals"])


def test_trace_shows_the_five_sections_for_a_judged_image(client):
    body = client.get("/api/operations/trace").json()
    assert body["empty"] is False and body["candidates"]
    image_id = body["candidates"][0]
    trace = client.get(f"/api/operations/trace/{image_id}").json()
    assert trace["known"] is True
    assert "출처" in trace["manifest"]
    assert trace["inferences"]["total"] >= 1
    _no_nan(trace)


def test_playback_renders_a_judged_video_as_a_job(client):
    source = video.make_sample(seconds=2, fps=10, size=(160, 120))
    listed = client.get("/api/operations/videos").json()
    assert str(source) in {v["path"] for v in listed["videos"]}
    assert client.get("/api/operations/videos/check", params={"path": str(source)}).json()["ok"]

    view = client.get("/api/operations/playback").json()
    assert view["kind"] == "anomaly"
    assert client.get(
        "/api/operations/playback/find", params={"source": str(source), "version": "v001"}
    ).json()["result"] is None

    job_id = client.post(
        "/api/operations/playback/render",
        json={"source": str(source), "version": "v001", "limit": 10},
    ).json()["job_id"]
    body = wait_job(client, job_id)
    assert body["status"] == "done", body["error"]
    result = body["result"]
    assert Path(result["video_path"]).is_file()
    assert result["frames"]["total"] == 10
    assert result["csv"].startswith("index")

    found = client.get(
        "/api/operations/playback/find", params={"source": str(source), "version": "v001"}
    ).json()["result"]
    assert found is not None and found["video_name"] == result["video_name"]
    download = client.get("/api/operations/playback/download", params={"path": result["video_path"]})
    assert download.status_code == 200
    stream = client.get("/api/operations/playback/video", params={"path": result["video_path"]})
    assert stream.status_code == 200 and stream.headers["content-type"].startswith("video/")
    assert client.get("/api/operations/playback/download", params={"path": "/etc/passwd"}).status_code == 404


def test_compare_needs_two_videos_with_truth(client):
    body = client.get("/api/operations/compare").json()
    assert body["empty"] is False
    assert body["names"] == []                          # 합성 데모에는 영상 프레임이 없다
    assert client.post(
        "/api/operations/compare/run", json={"left": "a", "right": "b", "fps": 30}
    ).status_code == 400


def test_clearing_the_log_empties_drift_and_scenario(client):
    removed = client.post("/api/operations/scenario/clear").json()["removed"]
    assert removed > 0
    client.post("/api/operations/inference/clear_log")
    assert client.get("/api/operations/inference").json()["log"]["summary"]["total"] == 0
    assert client.get("/api/operations/drift").json()["has_batch"] is False
    assert client.get("/api/operations/scenario").json()["result"] is None
