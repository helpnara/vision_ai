"""테스트 공통 준비.

경로가 프로젝트별로 갈리기 전에는 테스트마다 `DATA_ROOT`·`RAW_DIR`·`MANIFEST_PATH`…를
8줄씩 monkeypatch하고 있었다. 같은 8줄이 7개 파일에 흩어져 있어서, 경로가 하나 늘 때마다
7곳을 고쳐야 했다.

이제 경로는 **컨테이너 두 개**(`DATA_HOME`·`ARTIFACT_HOME`)에서 파생되므로 그 둘만
tmp_path로 돌리면 나머지가 따라온다.
"""

from __future__ import annotations

import pytest

from vision_ai import config


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """격리된 데이터·산출물 루트. 테스트는 서로의 파일을 보지 못한다."""
    monkeypatch.setattr(config, "DATA_HOME", tmp_path / "data")
    monkeypatch.setattr(config, "ARTIFACT_HOME", tmp_path / "artifacts")
    monkeypatch.setattr(config, "_active_project", config.DEFAULT_PROJECT)
    config.ensure_dirs()
    return config.data_root()
