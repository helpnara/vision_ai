"""2단계(라벨링) 코어 로직 테스트."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vision_ai import config, datasets, ingest, labeling, storage




def _write_image(path: Path, value: int = 150, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(value)
    image = np.clip(rng.normal(value, 10, (size, size, 3)), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


@pytest.fixture
def visa_like(sandbox):
    """VisA 구조의 소규모 데이터셋을 만들고 manifest에 등록한다."""
    root = sandbox / "VisA"
    for index, category in enumerate(("candle", "cashew")):
        for kind, count in (("Normal", 4), ("Anomaly", 2)):
            for i in range(count):
                # 카테고리·종류별로 픽셀값을 달리해 sha1 중복을 피한다
                seed = index * 100 + (0 if kind == "Normal" else 50) + i
                _write_image(root / category / "Data" / "Images" / kind / f"{i:04d}.JPG", 100 + seed % 120)
        # Anomaly 마스크 (첫 장만)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:35, 10:30] = 255
        mask_path = root / category / "Data" / "Masks" / "Anomaly" / "0000.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask)

    ingest.ingest_folder(root, source="visa", layout="visa")
    return root


# --- VisA 경로 파서 --------------------------------------------------------

@pytest.mark.parametrize(
    ("relative", "label", "defect_type", "is_mask"),
    [
        ("candle/Data/Images/Normal/0000.JPG", config.LABEL_NORMAL, config.DEFECT_TYPE_NONE, False),
        ("candle/Data/Images/Anomaly/000.JPG", config.LABEL_DEFECT, config.DEFECT_TYPE_UNSPECIFIED, False),
        ("candle/Data/Masks/Anomaly/000.png", config.LABEL_DEFECT, config.DEFECT_TYPE_UNSPECIFIED, True),
    ],
)
def test_parse_visa_path(relative, label, defect_type, is_mask):
    parsed = datasets.parse_visa_path(Path(relative))
    assert parsed["category"] == "candle"
    assert parsed["label"] == label
    assert parsed["defect_type"] == defect_type
    assert parsed["is_mask"] is is_mask
    # 분할은 split_csv에 정의되므로 폴더에서 추론하지 않는다
    assert parsed["split"] == config.SPLIT_NONE


def test_parse_visa_path_unknown_structure():
    parsed = datasets.parse_visa_path(Path("candle/whatever/0000.JPG"))
    assert parsed["label"] == config.LABEL_UNLABELED


def test_visa_is_default_dataset():
    default = datasets.default_dataset()
    assert default.key == "visa"
    assert default.layout == "visa"
    # 상업적 이용 가능이 선정 근거이므로 라이선스가 바뀌면 테스트가 알려주도록 둔다
    assert default.license == "CC BY 4.0"


def test_recommended_puts_default_first():
    assert datasets.recommended()[0].is_default


# --- 라벨 이벤트 / resolve -------------------------------------------------

def test_resolve_uses_folder_labels(visa_like):
    resolved = labeling.resolve()
    assert not resolved.empty
    assert set(resolved["label_source"]) == {"folder"}
    defects = resolved[resolved["label"] == config.LABEL_DEFECT]
    assert (defects["defect_type"] == config.DEFECT_TYPE_UNSPECIFIED).all()


def test_record_label_overrides_folder_label(visa_like):
    resolved = labeling.resolve()
    target = resolved[resolved["label"] == config.LABEL_DEFECT].iloc[0]

    labeling.record_label(
        target["image_id"], label=config.LABEL_DEFECT, defect_type="scratch", roi=(5, 6, 20, 30)
    )

    updated = labeling.resolve()
    row = updated[updated["image_id"] == target["image_id"]].iloc[0]
    assert row["defect_type"] == "scratch"
    assert row["label_source"] == "human"
    assert bool(row["verified"]) is True
    assert (row["roi_x"], row["roi_y"], row["roi_w"], row["roi_h"]) == (5, 6, 20, 30)


def test_latest_event_wins(visa_like):
    resolved = labeling.resolve()
    image_id = resolved.iloc[0]["image_id"]

    labeling.record_label(image_id, label=config.LABEL_DEFECT, defect_type="crack")
    labeling.record_label(image_id, label=config.LABEL_NORMAL)

    row = labeling.resolve().query("image_id == @image_id").iloc[0]
    assert row["label"] == config.LABEL_NORMAL
    # 이력은 지워지지 않는다
    assert len(labeling.load_events()) == 2


def test_relabel_normal_to_defect(visa_like):
    resolved = labeling.resolve()
    target = resolved[resolved["label"] == config.LABEL_NORMAL].iloc[0]
    labeling.record_label(target["image_id"], label=config.LABEL_DEFECT, defect_type="stain")

    row = labeling.resolve().query("image_id == @target.image_id").iloc[0]
    assert row["label"] == config.LABEL_DEFECT
    assert row["defect_type"] == "stain"


def test_resolve_on_empty_manifest(sandbox):
    resolved = labeling.resolve()
    assert resolved.empty
    assert labeling.stats(resolved)["total"] == 0


# --- 결함 유형 정규화 ------------------------------------------------------

def test_normalize_defect_type_uses_default_map():
    mapping = labeling.load_type_map()
    assert labeling.normalize_defect_type("broken_large", mapping) == "broken"
    assert labeling.normalize_defect_type("bent_lead", mapping) == "deformation"


def test_normalize_keeps_unknown_type_visible():
    """모르는 유형을 other로 뭉개면 사람이 매핑해야 한다는 사실이 숨는다."""
    mapping = labeling.load_type_map()
    assert labeling.normalize_defect_type("weird_new_type", mapping) == "weird_new_type"


def test_normalize_passes_through_standard_types():
    mapping = labeling.load_type_map()
    for key in config.DEFECT_TYPES:
        assert labeling.normalize_defect_type(key, mapping) == key
    assert labeling.normalize_defect_type(config.DEFECT_TYPE_UNSPECIFIED, mapping) == (
        config.DEFECT_TYPE_UNSPECIFIED
    )


def test_save_and_load_type_map(sandbox):
    labeling.save_type_map({"my_raw": "scratch"})
    mapping = labeling.load_type_map()
    assert mapping["my_raw"] == "scratch"
    assert mapping["broken_large"] == "broken"  # 기본 매핑도 유지된다


def test_unmapped_defect_types(sandbox):
    manifest = pd.DataFrame(
        [
            {"image_id": "a", "path": "x.png", "source": "s", "category": "c",
             "split": config.SPLIT_NONE, "label": config.LABEL_DEFECT, "defect_type": "zzz_unknown"},
            {"image_id": "b", "path": "y.png", "source": "s", "category": "c",
             "split": config.SPLIT_NONE, "label": config.LABEL_DEFECT, "defect_type": "scratch"},
        ]
    )
    resolved = labeling.resolve(manifest)
    assert labeling.unmapped_defect_types(resolved) == ["zzz_unknown"]

    labeling.save_type_map({"zzz_unknown": "other"})
    assert labeling.unmapped_defect_types(labeling.resolve(manifest)) == []


# --- ROI / 마스크 ----------------------------------------------------------

def test_find_mask_path_visa(visa_like):
    image = visa_like / "candle" / "Data" / "Images" / "Anomaly" / "0000.JPG"
    mask = labeling.find_mask_path(image)
    assert mask is not None
    assert mask.name == "0000.png"
    assert "Masks" in mask.parts


def test_find_mask_path_returns_none_for_normal(visa_like):
    image = visa_like / "candle" / "Data" / "Images" / "Normal" / "0000.JPG"
    assert labeling.find_mask_path(image) is None


def test_find_mask_path_mvtec_style(tmp_path):
    image = tmp_path / "wood" / "test" / "scratch" / "000.png"
    _write_image(image)
    mask = tmp_path / "wood" / "ground_truth" / "scratch" / "000_mask.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask), np.zeros((64, 64), dtype=np.uint8))
    assert labeling.find_mask_path(image) == mask


def test_roi_from_mask(tmp_path):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:41, 10:31] = 255
    path = tmp_path / "m.png"
    cv2.imwrite(str(path), mask)
    assert labeling.roi_from_mask(path) == (10, 20, 21, 21)


def test_roi_from_empty_mask_is_none(tmp_path):
    path = tmp_path / "empty.png"
    cv2.imwrite(str(path), np.zeros((32, 32), dtype=np.uint8))
    assert labeling.roi_from_mask(path) is None


def test_roi_from_image_path(visa_like):
    resolved = labeling.resolve()
    defects = resolved[resolved["label"] == config.LABEL_DEFECT]
    found = [labeling.roi_from_image_path(p) for p in defects["path"]]
    # 카테고리마다 0000 한 장에만 마스크를 만들었으므로 일부는 None이다
    assert any(r == (10, 20, 20, 15) for r in found), found


# --- 분할 -----------------------------------------------------------------

def test_allocate_sums_to_total():
    for total in (0, 1, 2, 3, 7, 10, 101):
        counts = labeling._allocate(total, (0.6, 0.2, 0.2))
        assert sum(counts) == total
        assert all(c >= 0 for c in counts)


def test_assign_splits_is_stratified(visa_like):
    resolved = labeling.resolve()
    mapping = labeling.assign_splits(resolved, train=0.5, val=0.25, test=0.25, seed=1)
    assert len(mapping) == len(resolved)

    labeling.save_splits(mapping)
    updated = labeling.resolve()
    # 각 (카테고리, 라벨) 조합이 한 분할에만 몰리지 않는다
    grouped = updated.groupby(["category", "label"])["split"].nunique()
    assert (grouped >= 2).all(), updated[["category", "label", "split"]].to_string()


def test_assign_splits_is_deterministic(visa_like):
    resolved = labeling.resolve()
    first = labeling.assign_splits(resolved, seed=7)
    second = labeling.assign_splits(resolved, seed=7)
    assert first == second
    assert labeling.assign_splits(resolved, seed=8) != first


def test_assign_splits_skips_unlabeled(sandbox):
    manifest = pd.DataFrame(
        [
            {"image_id": str(i), "path": f"{i}.png", "source": "s", "category": "c",
             "split": config.SPLIT_NONE, "label": config.LABEL_UNLABELED,
             "defect_type": config.DEFECT_TYPE_NONE}
            for i in range(4)
        ]
    )
    resolved = labeling.resolve(manifest)
    assert labeling.assign_splits(resolved, labeled_only=True) == {}
    assert len(labeling.assign_splits(resolved, labeled_only=False)) == 4


def test_clear_splits(visa_like):
    resolved = labeling.resolve()
    labeling.save_splits(labeling.assign_splits(resolved))
    assert labeling.load_splits()
    labeling.clear_splits()
    assert labeling.load_splits() == {}


def _visa_csv_rows(manifest: pd.DataFrame, split: str = "train") -> list[dict]:
    """VisA `split_csv/1cls.csv`를 흉내낸 행을 만든다.

    CSV는 데이터셋 루트 기준 경로(`candle/Data/...`)를, manifest는 데이터 루트 기준
    경로(`VisA/candle/Data/...`)를 담으므로 앞부분이 다르다.
    """
    rows = []
    for path_value in manifest["path"].astype(str):
        parts = Path(path_value).parts
        rows.append(
            {
                "object": parts[1] if len(parts) > 1 else "candle",
                "split": split,
                "label": "normal",
                "image": "/".join(parts[1:]),  # 선행 "VisA/" 제거
            }
        )
    return rows


def test_import_split_csv(visa_like, tmp_path):
    manifest = storage.load_manifest()
    csv_path = tmp_path / "1cls.csv"
    pd.DataFrame(_visa_csv_rows(manifest)).to_csv(csv_path, index=False)

    mapping, unmatched = labeling.import_split_csv(csv_path, manifest)
    assert unmatched == 0
    assert set(mapping.values()) == {config.SPLIT_TRAIN}
    assert len(mapping) == len(manifest)


def test_import_split_csv_distinguishes_reused_filenames(visa_like, tmp_path):
    """VisA는 Normal/Anomaly에 같은 파일명을 쓴다 — 파일명만으로는 구분할 수 없다."""
    manifest = storage.load_manifest()
    names = manifest["path"].astype(str).map(lambda p: Path(p).name)
    assert names.duplicated().any(), "파일명 중복이 없으면 이 테스트가 의미 없다"

    rows = _visa_csv_rows(manifest)
    # 결함 이미지만 test로 지정해, 경로로 정확히 구분되는지 확인한다
    for row in rows:
        row["split"] = "test" if "Anomaly" in row["image"] else "train"
    csv_path = tmp_path / "2cls.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    mapping, unmatched = labeling.import_split_csv(csv_path, manifest)
    assert unmatched == 0

    labeling.save_splits(mapping)
    resolved = labeling.resolve()
    defects = resolved[resolved["label"] == config.LABEL_DEFECT]
    normals = resolved[resolved["label"] == config.LABEL_NORMAL]
    assert set(defects["split"]) == {config.SPLIT_TEST}
    assert set(normals["split"]) == {config.SPLIT_TRAIN}


def test_import_split_csv_handles_windows_separators(visa_like, tmp_path):
    manifest = storage.load_manifest()
    rows = _visa_csv_rows(manifest)
    for row in rows:
        row["image"] = row["image"].replace("/", "\\")
    csv_path = tmp_path / "win.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    mapping, unmatched = labeling.import_split_csv(csv_path, manifest)
    assert unmatched == 0
    assert len(mapping) == len(manifest)


def test_import_split_csv_reports_unmatched(visa_like, tmp_path):
    csv_path = tmp_path / "other.csv"
    pd.DataFrame(
        [{"image": "nope/does/not/exist.JPG", "split": "train"},
         {"image": "also/missing.JPG", "split": "bogus_split"}]
    ).to_csv(csv_path, index=False)

    mapping, unmatched = labeling.import_split_csv(csv_path, storage.load_manifest())
    assert mapping == {}
    assert unmatched == 2


def test_import_split_csv_rejects_bad_columns(visa_like, tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([{"a": 1, "b": 2}]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="컬럼"):
        labeling.import_split_csv(csv_path, storage.load_manifest())


# --- 검수 큐 / 통계 --------------------------------------------------------

def test_review_queue_unspecified(visa_like):
    resolved = labeling.resolve()
    queue = labeling.review_queue(resolved, mode="unspecified")
    assert len(queue) == int((resolved["label"] == config.LABEL_DEFECT).sum())
    assert (queue["defect_type"] == config.DEFECT_TYPE_UNSPECIFIED).all()


def test_review_queue_drains_after_labeling(visa_like):
    resolved = labeling.resolve()
    before = len(labeling.review_queue(resolved, mode="unspecified"))
    target = labeling.review_queue(resolved, mode="unspecified").iloc[0]
    labeling.record_label(target["image_id"], label=config.LABEL_DEFECT, defect_type="crack")
    after = len(labeling.review_queue(labeling.resolve(), mode="unspecified"))
    assert after == before - 1


def test_review_queue_filters_by_category(visa_like):
    resolved = labeling.resolve()
    queue = labeling.review_queue(resolved, mode="all", category="candle")
    assert set(queue["category"]) == {"candle"}


def test_review_queue_unverified(visa_like):
    resolved = labeling.resolve()
    total = len(resolved)
    assert len(labeling.review_queue(resolved, mode="unverified")) == total
    labeling.record_label(resolved.iloc[0]["image_id"], label=config.LABEL_NORMAL)
    assert len(labeling.review_queue(labeling.resolve(), mode="unverified")) == total - 1


def test_stats_tracks_progress(visa_like):
    resolved = labeling.resolve()
    stats = labeling.stats(resolved)
    assert stats["total"] == len(resolved)
    assert stats["human"] == 0
    assert stats["unspecified_type"] == stats["defect"]
    assert stats["split_assigned"] == 0

    target = resolved[resolved["label"] == config.LABEL_DEFECT].iloc[0]
    labeling.record_label(target["image_id"], label=config.LABEL_DEFECT,
                          defect_type="scratch", roi=(1, 2, 3, 4))
    stats = labeling.stats(labeling.resolve())
    assert stats["human"] == 1
    assert stats["verified"] == 1
    assert stats["with_roi"] == 1
    assert stats["unspecified_type"] == stats["defect"] - 1


# --- 그룹 인지 분할 (H2) ----------------------------------------------------

def _frames(video_count=3, per_video=8):
    """영상 여러 개에서 뽑은 프레임을 흉내 낸다."""
    rows = []
    for v in range(video_count):
        for f in range(per_video):
            rows.append({
                "image_id": f"v{v}f{f}", "category": "cctv",
                "label": config.LABEL_NORMAL if f % 2 else config.LABEL_DEFECT,
                "group": f"video{v}", "ingested_at": f"2026-01-0{v+1}T00:00:{f:02d}",
            })
    return pd.DataFrame(rows)


def test_frames_from_one_video_never_straddle_splits():
    """이 테스트가 그룹 분할의 존재 이유다.

    같은 영상의 프레임은 서로 너무 비슷해서, 학습과 평가에 나뉘어 들어가면 성능이
    실제보다 높게 나온다. AUROC 0.99인데 현장에서 무너지는 모델이 그렇게 나온다.
    """
    frame = _frames()
    mapping = labeling.assign_splits(frame, mode=labeling.SPLIT_BY_GROUP, seed=3)
    frame = frame.assign(split=frame["image_id"].map(mapping))
    per_group = frame.groupby(["group", "label"])["split"].nunique()
    assert (per_group == 1).all(), frame[["group", "label", "split"]].to_string()


def test_random_mode_does_straddle_splits():
    """무작위가 왜 위험한지 확인해 둔다 — 고르는 사람이 알고 골라야 한다."""
    frame = _frames()
    mapping = labeling.assign_splits(frame, mode=labeling.SPLIT_BY_RANDOM, seed=3)
    frame = frame.assign(split=frame["image_id"].map(mapping))
    per_group = frame.groupby("group")["split"].nunique()
    assert (per_group > 1).any()


def test_time_mode_puts_early_frames_in_train():
    """영상이 하나뿐이라 그룹으로 못 나눌 때 쓴다. 앞은 학습, 뒤는 평가."""
    frame = _frames(video_count=1, per_video=20)
    mapping = labeling.assign_splits(frame, mode=labeling.SPLIT_BY_TIME, seed=1)
    ordered = [mapping[i] for i in frame.sort_values("ingested_at")["image_id"]]
    assert ordered[0] == config.SPLIT_TRAIN
    assert ordered[-1] != config.SPLIT_TRAIN


def test_groups_are_split_by_image_count_not_group_count():
    """영상마다 프레임 수가 크게 다르다. 개수로 나누면 이미지가 9:1:1이 되기도 한다."""
    rows = []
    for name, count in (("big", 40), ("small1", 2), ("small2", 2), ("small3", 2)):
        for f in range(count):
            rows.append({
                "image_id": f"{name}-{f}", "category": "c", "label": config.LABEL_NORMAL,
                "group": name,
            })
    frame = pd.DataFrame(rows)
    mapping = labeling.assign_splits(frame, mode=labeling.SPLIT_BY_GROUP, seed=5)
    counts = pd.Series(mapping).value_counts()
    assert counts.get(config.SPLIT_TRAIN, 0) >= counts.get(config.SPLIT_TEST, 0)


def test_missing_group_column_behaves_like_before():
    """예전 데이터는 group 열이 없다. 이미지 하나가 곧 그룹이라 동작이 그대로여야 한다."""
    frame = _frames().drop(columns=["group"])
    mapping = labeling.assign_splits(frame, mode=labeling.SPLIT_BY_GROUP, seed=2)
    assert len(set(mapping.values())) > 1


def test_blank_group_falls_back_to_the_image_itself():
    frame = _frames()
    frame.loc[frame.index[:4], "group"] = ""
    groups = labeling.image_groups(frame)
    assert list(groups[:4]) == list(frame["image_id"][:4])


def test_every_split_gets_something_when_there_are_enough_groups():
    """평가 분할이 비면 3단계 학습 자체가 막힌다."""
    mapping = labeling.assign_splits(_frames(video_count=6), mode=labeling.SPLIT_BY_GROUP, seed=1)
    assert set(mapping.values()) == {config.SPLIT_TRAIN, config.SPLIT_VAL, config.SPLIT_TEST}


def test_a_single_group_stays_in_train():
    frame = _frames(video_count=1, per_video=4)
    mapping = labeling.assign_splits(frame, mode=labeling.SPLIT_BY_GROUP, seed=1)
    assert set(mapping.values()) == {config.SPLIT_TRAIN}


def test_unknown_split_mode_is_refused():
    with pytest.raises(ValueError):
        labeling.assign_splits(_frames(), mode="없는방식")
