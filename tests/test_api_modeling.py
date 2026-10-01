"""3단계 웹 API — 학습 데이터 · 베이스라인 · 이상탐지 · Claude 2차 판정 · 평가 리포트 · 실험 기록.

코어 로직은 `test_modeling.py`가 본다. 여기서는 **배선**만 본다: 라우터가 코어를 바르게
부르고, 응답이 JSON으로 나가며(NaN 없음), 잡이 끝까지 돌아 결과를 상태에 남기는가.
"""

from __future__ import annotations

import json
import math
import time

import pytest
from fastapi.testclient import TestClient

from vision_ai import config, ingest, labeling


@pytest.fixture
def client(sandbox):
    from server import state
    from server.main import app

    state.clear()
    return TestClient(app)


@pytest.fixture
def demo(client):
    """합성 샘플 2종 + 분할. 베이스라인(결함 ≥ 10)과 이상탐지(정상 > 14) 둘 다 가능한 크기."""
    out = ingest.generate_synthetic(
        categories=["pcb_green", "metal"], n_normal=24, n_defect=12, size=96, seed=7, layout="visa"
    )
    ingest.ingest_folder(out, source="synthetic", layout="visa")
    resolved = labeling.resolve()
    labeling.save_splits(labeling.assign_splits(resolved, train=0.6, val=0.2, test=0.2, seed=3))
    return client


def wait_job(client: TestClient, job_id: str, timeout: int = 180) -> dict:
    for _ in range(timeout * 10):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("잡이 끝나지 않았다")


def _no_nan(value) -> None:
    """응답 트리 어디에도 NaN/inf가 없어야 한다 — JSON이 깨진다."""
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            _no_nan(item)
    elif isinstance(value, list):
        for item in value:
            _no_nan(item)


# --- 비어 있을 때 ---------------------------------------------------------------

def test_empty_project_points_to_ingest(client):
    body = client.get("/api/modeling/data").json()
    assert body["empty"] is True
    assert body["link"]["route"] == "/ingest"

    assert client.get("/api/modeling/baseline").json()["empty"] is True
    assert client.get("/api/modeling/anomaly").json()["empty"] is True
    assert client.get("/api/modeling/report").json() == {"has_result": False}
    assert client.get("/api/modeling/experiments").json() == {"empty": True}
    assert client.post("/api/modeling/data/extract").status_code == 400


def test_claude_tab_answers_without_a_key_or_result(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    body = client.get("/api/modeling/claude").json()
    assert body["has_result"] is False
    assert "ANTHROPIC_API_KEY" in body["hint"]
    assert body["reviews"]["empty"] is True
    assert client.post("/api/modeling/claude/run", json={"limit": 3}).status_code == 400


# --- 학습 데이터 ------------------------------------------------------------------

def test_data_tab_describes_the_training_set(demo):
    body = demo.get("/api/modeling/data").json()
    assert body["empty"] is False
    assert body["ready"] is True and body["problems"] == []
    assert body["synthetic"] is True
    assert body["metrics"]["labeled"] == body["metrics"]["normal"] + body["metrics"]["defect"]
    assert body["metrics"]["categories"] == 2
    assert set(body["crosstab"]["columns"]) >= {"split", "정상", "결함"}
    assert body["features"]["count"] == len(body["features"]["names"])
    assert body["dataset"] is None

    job = wait_job(demo, demo.post("/api/modeling/data/extract").json()["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["result"]["rows"] == body["metrics"]["labeled"]
    assert "특징 준비 완료" in job["result"]["done"]
    assert demo.get("/api/modeling/data").json()["dataset"]["rows"] == body["metrics"]["labeled"]


# --- 베이스라인 ------------------------------------------------------------------

def test_baseline_job_records_a_run_and_fills_the_report(demo):
    tab = demo.get("/api/modeling/baseline").json()
    assert tab["advice"]["blocked"] is False
    assert tab["ready"] is True
    assert tab["show_categories"] is True
    assert tab["target_recall"] == pytest.approx(0.95)
    assert tab["counts"]["train"] > 0 and tab["counts"][config.SPLIT_TEST] > 0

    started = demo.post("/api/modeling/baseline/train", json={
        "kind": "logreg", "balanced": True, "eval_split": "test", "target_recall": 0.9, "use_categories": False,
    })
    assert started.status_code == 200, started.text
    job = wait_job(demo, started.json()["job_id"])
    assert job["status"] == "done", job["error"]
    result = job["result"]
    _no_nan(result)
    assert result["kind"] == "baseline"
    assert {m["key"] for m in result["metric_row"]} == {"recall", "precision", "f1", "auroc", "miss_fp"}
    assert all(m["help"] for m in result["metric_row"])
    assert len(result["importance"]) == 15
    # 카테고리가 둘이므로 스위치를 껐어도 표는 보인다.
    assert result["categories"] is not None and result["categories"]["enabled"] is False
    assert result["categories"]["table"]["total"] == 2

    # 실험 기록에 남고, 탭을 다시 열어도 마지막 결과가 보인다.
    runs = demo.get("/api/modeling/experiments").json()
    assert result["run_id"] in runs["run_ids"]
    assert demo.get(f"/api/modeling/experiments/{result['run_id']}").json()["kind"] == "baseline"
    assert demo.get("/api/modeling/experiments/nope").status_code == 404
    assert demo.get("/api/modeling/baseline").json()["last"]["run_id"] == result["run_id"]
    assert demo.get("/api/modeling/anomaly").json()["last"] is None

    report = demo.get("/api/modeling/report").json()
    assert report["has_result"] is True
    assert report["slider"]["min"] <= report["slider"]["value"] <= report["slider"]["max"]


def test_baseline_rejects_bad_choices(demo):
    assert demo.post("/api/modeling/baseline/train", json={"kind": "svm"}).status_code == 400
    assert demo.post("/api/modeling/baseline/train", json={"eval_split": "train"}).status_code == 400


# --- 이상탐지 ---------------------------------------------------------------------

def test_anomaly_job_enables_heatmaps(demo):
    tab = demo.get("/api/modeling/anomaly").json()
    assert tab["advice"]["blocked"] is False
    assert tab["heatmap"]["available"] is False
    assert tab["counts"]["train_normal"] > tab["patch_feature_dim"]
    assert tab["cnn"]["available"] in (True, False)

    started = demo.post("/api/modeling/anomaly/train", json={
        "eval_split": "test", "backend": "classic", "per_position": True, "score_mode": "p99",
        "target_recall": 0.9, "use_categories": True,
    })
    assert started.status_code == 200, started.text
    job = wait_job(demo, started.json()["job_id"])
    assert job["status"] == "done", job["error"]
    result = job["result"]
    _no_nan(result)
    assert result["kind"] == "anomaly"
    assert result["cache_note"] and "새로 뽑아" in result["cache_note"]
    assert result["categories"]["enabled"] is True

    tab = demo.get("/api/modeling/anomaly").json()
    assert tab["heatmap"]["available"] is True
    image_id = tab["heatmap"]["defects"][0]
    for suffix in ("input", "heatmap", "overlay"):
        got = demo.get(f"/api/modeling/anomaly/heatmap/{image_id}/{suffix}")
        assert got.status_code == 200
        assert got.headers["content-type"] == "image/jpeg"
    info = demo.get(f"/api/modeling/anomaly/heatmap/{image_id}").json()
    _no_nan(info)
    assert info["image_id"] == image_id
    assert "localization" in info
    assert demo.get("/api/modeling/anomaly/heatmap/ghost").status_code == 404

    # 두 번째 실행은 저장해 둔 격자를 그대로 쓴다.
    job = wait_job(demo, demo.post("/api/modeling/anomaly/train", json={"target_recall": 0.9}).json()["job_id"])
    assert job["status"] == "done", job["error"]
    assert "그대로 썼습니다" in job["result"]["cache_note"]


def test_heatmap_requires_a_trained_model(demo):
    assert demo.get("/api/modeling/anomaly/heatmap/x/input").status_code == 404


# --- 평가 리포트 · Claude ---------------------------------------------------------

@pytest.fixture
def trained(demo):
    job = wait_job(demo, demo.post("/api/modeling/baseline/train", json={"target_recall": 0.9}).json()["job_id"])
    assert job["status"] == "done", job["error"]
    return demo


def test_report_evaluate_is_json_safe_and_complete(trained):
    report = trained.get("/api/modeling/report").json()
    body = trained.post("/api/modeling/report/evaluate", json={
        "threshold": report["slider"]["value"], "prevalence": 0.01, "volume": 1000, "outcome": "FN",
    })
    assert body.status_code == 200, body.text
    data = body.json()
    _no_nan(data)
    assert data["verdict"]["level"] in ("good", "usable", "weak")
    assert data["verdict"]["headline"]
    assert data["confusion"]["total"] == 2
    assert data["sweep"]["$schema"].startswith("https://vega.github.io/")
    assert data["roc"] is not None
    assert data["impact"]["volume"] == 1000
    assert data["impact_by_prevalence"]["total"] == 4
    assert data["subset_count"] == data["subset"]["total"]
    assert len(data["samples"]) <= 8
    assert all("image_id" in s for s in data["samples"])

    files = trained.post("/api/modeling/report/files", json={"threshold": report["slider"]["value"]}).json()
    assert files["report"]["filename"].endswith(".md")
    assert "표면 결함 탐지 결과 보고서" in files["report"]["text"]
    assert "합성 샘플" in files["report"]["text"]
    assert files["errors"]["text"].startswith("image_id,")
    assert trained.post("/api/modeling/report/evaluate", json={"threshold": 0.5, "outcome": "XX"}).status_code == 400


def test_claude_tab_selects_uncertain_candidates(trained, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    body = trained.get("/api/modeling/claude?limit=4").json()
    assert body["has_result"] is True
    assert 0 < body["candidates"]["count"] <= 4
    assert body["candidates"]["cost"] > 0
    assert body["candidates"]["table"]["columns"] == ["image_id", "label", "score", "outcome", "uncertainty"]
    assert trained.post("/api/modeling/claude/run", json={"limit": 2, "effort": "extreme"}).status_code == 400
    _no_nan(body)
    json.dumps(body)
