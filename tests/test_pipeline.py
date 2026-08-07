"""1단계(데이터 수집) 코어 로직 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vision_ai import config, datasets, ingest, quality, storage




# --- 경로 파서 -------------------------------------------------------------

@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("bottle/train/good/000.png", ("bottle", "train", config.LABEL_NORMAL, "none")),
        ("bottle/test/good/001.png", ("bottle", "test", config.LABEL_NORMAL, "none")),
        ("wood/test/scratch/002.png", ("wood", "test", config.LABEL_DEFECT, "scratch")),
        ("zipper/validation/good/003.png", ("zipper", "val", config.LABEL_NORMAL, "none")),
    ],
)
def test_parse_mvtec_path(relative, expected):
    parsed = datasets.parse_mvtec_path(Path(relative))
    assert (parsed["category"], parsed["split"], parsed["label"], parsed["defect_type"]) == expected
    assert parsed["is_mask"] is False


def test_parse_mvtec_path_detects_mask():
    parsed = datasets.parse_mvtec_path(Path("wood/ground_truth/scratch/000_mask.png"))
    assert parsed["is_mask"] is True


def test_parse_mvtec_path_handles_unknown_structure():
    parsed = datasets.parse_mvtec_path(Path("stray.png"))
    assert parsed["label"] == config.LABEL_UNLABELED


def test_parse_flat_path():
    defective = datasets.parse_flat_path(Path("MT_Crack/exp1.jpg"))
    assert defective["label"] == config.LABEL_DEFECT
    assert defective["defect_type"] == "MT_Crack"

    normal = datasets.parse_flat_path(Path("MT_Free/exp2.jpg"))
    assert normal["label"] == config.LABEL_NORMAL


def test_parse_path_custom_layout_is_unlabeled():
    parsed = datasets.parse_path(Path("a/b/c.png"), "custom")
    assert parsed["label"] == config.LABEL_UNLABELED


# --- 카탈로그 -------------------------------------------------------------

def test_catalog_keys_unique():
    keys = datasets.keys()
    assert len(keys) == len(set(keys))


def test_catalog_entries_are_wellformed():
    for dataset in datasets.CATALOG:
        assert dataset.name and dataset.url
        assert dataset.layout in datasets.LAYOUT_OPTIONS
        assert 1 <= dataset.everyday_fit <= 5


def test_recommended_is_sorted_by_everyday_fit():
    fits = [d.everyday_fit for d in datasets.recommended()]
    assert fits == sorted(fits, reverse=True)


# --- 품질 점검 -------------------------------------------------------------

def test_assess_array_flags_low_resolution():
    tiny = np.full((10, 10, 3), 128, dtype=np.uint8)
    result = quality.assess_array(tiny)
    assert "low_resolution" in result.flags
    assert result.is_usable is False


def test_assess_array_flags_flat_image_as_blurry_and_dark():
    dark = np.zeros((128, 128, 3), dtype=np.uint8)
    result = quality.assess_array(dark)
    assert "blurry" in result.flags
    assert "underexposed" in result.flags
    assert result.width == 128 and result.height == 128


def test_decode_rejects_non_image():
    assert quality.decode(b"not an image") is None


def test_describe_flags():
    assert quality.describe_flags("") == "양호"
    assert "흐림" in quality.describe_flags("blurry")


# --- 수집 + manifest ------------------------------------------------------

def test_generate_synthetic_creates_mvtec_layout(sandbox):
    out = ingest.generate_synthetic(
        categories=["wood_panel"], n_normal=5, n_defect=4, size=128, seed=7, layout="mvtec"
    )
    assert (out / "wood_panel" / "train" / "good").is_dir()
    assert (out / "wood_panel" / "test" / "scratch").is_dir()
    assert ingest.count_image_files(out) > 0


def test_generate_synthetic_defaults_to_visa_layout(sandbox):
    """기본 예시 데이터셋이 VisA이므로 합성 데이터도 같은 구조로 나와야 한다."""
    out = ingest.generate_synthetic(
        categories=["wood_panel"], n_normal=5, n_defect=4, size=128, seed=7
    )
    assert (out / "wood_panel" / "Data" / "Images" / "Normal").is_dir()
    assert (out / "wood_panel" / "Data" / "Images" / "Anomaly").is_dir()
    assert (out / "wood_panel" / "Data" / "Masks" / "Anomaly").is_dir()


def test_generate_synthetic_rejects_unknown_layout(sandbox):
    with pytest.raises(ValueError, match="layout"):
        ingest.generate_synthetic(categories=["fabric"], layout="nope")


@pytest.mark.parametrize("layout", ["visa", "mvtec"])
def test_synthetic_masks_pair_with_defect_images(sandbox, layout):
    """마스크는 결함 이미지와 1:1로 대응해야 한다 (ROI 자동 추출의 전제)."""
    from vision_ai import labeling

    out = ingest.generate_synthetic(
        categories=["painted_metal"], n_normal=4, n_defect=4, size=128, seed=9, layout=layout
    )
    ingest.ingest_folder(out, source="synthetic", layout=layout)

    resolved = labeling.resolve()
    defects = resolved[resolved["label"] == config.LABEL_DEFECT]
    assert not defects.empty

    rois = [labeling.roi_from_image_path(p) for p in defects["path"]]
    assert all(r is not None for r in rois), f"{layout}: 마스크를 찾지 못한 결함 이미지가 있다"
    for x, y, w, h in rois:
        assert w > 0 and h > 0
        assert 0 <= x < 128 and 0 <= y < 128


def test_synthetic_normal_images_have_no_mask(sandbox):
    from vision_ai import labeling

    out = ingest.generate_synthetic(
        categories=["fabric"], n_normal=4, n_defect=4, size=128, seed=11
    )
    ingest.ingest_folder(out, source="synthetic", layout="visa")

    resolved = labeling.resolve()
    normals = resolved[resolved["label"] == config.LABEL_NORMAL]
    assert not normals.empty
    assert all(labeling.roi_from_image_path(p) is None for p in normals["path"])


@pytest.mark.parametrize("defect", ingest.SYNTHETIC_DEFECTS)
def test_draw_defect_mask_matches_drawing(defect):
    """마스크가 실제 변경 영역을 감싸야 한다 — 어긋나면 ROI 정답이 틀린다."""
    rng = np.random.default_rng(3)
    surface = ingest._make_surface(rng, ingest.SURFACE_STYLES["ceramic_plate"], 256)
    marked, mask = ingest._draw_defect(np.random.default_rng(5), surface, defect)

    assert mask.shape == surface.shape[:2]
    assert mask.dtype == np.uint8
    assert (mask > 0).any(), "마스크가 비어 있다"

    changed = np.abs(marked.astype(int) - surface.astype(int)).max(axis=2) > 3
    ys, xs = np.nonzero(changed)
    my, mx = np.nonzero(mask > 0)
    # 마스크 바운딩박스가 실제 변경 영역을 (여유 5px 안에서) 포함한다
    assert mx.min() <= xs.min() + 5 and mx.max() >= xs.max() - 5
    assert my.min() <= ys.min() + 5 and my.max() >= ys.max() - 5


@pytest.mark.parametrize("defect", ingest.SYNTHETIC_DEFECTS)
def test_draw_defect_stays_local(defect):
    """결함은 국소 현상이어야 한다 — 전체 블렌딩으로 주변까지 뭉개면 안 된다."""
    rng = np.random.default_rng(0)
    style = ingest.SURFACE_STYLES["painted_metal"]
    surface = ingest._make_surface(rng, style, 256)

    marked, mask = ingest._draw_defect(np.random.default_rng(1), surface, defect)

    changed = (np.abs(marked.astype(int) - surface.astype(int)).max(axis=2) > 3)
    assert changed.any(), "결함이 전혀 그려지지 않았다"
    assert changed.mean() < 0.25, f"{defect}가 이미지 {changed.mean():.0%}를 변경했다 (국소성 위반)"


@pytest.mark.parametrize("category", list(ingest.SURFACE_STYLES))
def test_synthetic_surfaces_pass_quality_check(category):
    """합성 표면이 품질 경고를 유발하지 않아야 한다 (임계값과 생성 파라미터 정합성)."""
    rng = np.random.default_rng(4)
    surface = ingest._make_surface(rng, ingest.SURFACE_STYLES[category], 256)
    assert quality.assess_array(surface).flags == (), f"{category} 표면에 품질 경고 발생"


def test_ingest_folder_registers_and_dedupes(sandbox):
    out = ingest.generate_synthetic(
        categories=["wood_panel"], n_normal=5, n_defect=4, size=128, seed=7, layout="mvtec"
    )

    first = ingest.ingest_folder(out, source="synthetic", layout="mvtec")
    assert first.added > 0
    assert first.duplicates == 0

    df = storage.load_manifest()
    assert len(df) == first.added
    assert set(df["label"]) <= {config.LABEL_NORMAL, config.LABEL_DEFECT}
    assert (df["source"] == "synthetic").all()

    # 같은 폴더를 다시 등록하면 전부 중복으로 걸러진다
    second = ingest.ingest_folder(out, source="synthetic", layout="mvtec")
    assert second.added == 0
    assert second.duplicates == first.added
    assert len(storage.load_manifest()) == first.added


def test_manifest_paths_are_relative_and_resolvable(sandbox):
    out = ingest.generate_synthetic(
        categories=["fabric"], n_normal=3, n_defect=4, size=128, seed=3
    )
    ingest.ingest_folder(out, source="synthetic", layout="visa")

    df = storage.load_manifest()
    for value in df["path"]:
        assert not Path(value).is_absolute()
        assert storage.resolve_path(value).exists()


def test_ingest_uploads(sandbox):
    import cv2

    image = np.full((128, 128, 3), 180, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    payload = [("photo.png", buffer.tobytes()), ("broken.png", b"not an image")]

    result = ingest.ingest_uploads(
        payload, category="mug", label=config.LABEL_DEFECT, defect_type="scratch"
    )
    assert result.added == 1
    assert result.failed == ["broken.png"]

    df = storage.load_manifest()
    assert df.iloc[0]["category"] == "mug"
    assert df.iloc[0]["label"] == config.LABEL_DEFECT
    assert df.iloc[0]["defect_type"] == "scratch"


def test_remove_source(sandbox):
    out = ingest.generate_synthetic(
        categories=["ceramic_plate"], n_normal=3, n_defect=4, size=128, seed=5
    )
    ingest.ingest_folder(out, source="synthetic", layout="visa")
    assert len(storage.load_manifest()) > 0

    removed = ingest.remove_source("synthetic")
    assert removed > 0
    assert storage.load_manifest().empty


def test_summarize_empty_manifest(sandbox):
    stats = storage.summarize(storage.empty_manifest())
    assert stats["total"] == 0
    assert stats["defect"] == 0


def test_summarize_counts_labels(sandbox):
    out = ingest.generate_synthetic(
        categories=["wood_panel"], n_normal=10, n_defect=8, size=128, seed=11
    )
    ingest.ingest_folder(out, source="synthetic", layout="visa")

    stats = storage.summarize(storage.load_manifest())
    assert stats["total"] == stats["normal"] + stats["defect"]
    assert stats["normal"] > 0 and stats["defect"] > 0
    assert stats["categories"] == 1


def test_ingest_folder_missing_directory_raises(sandbox, tmp_path):
    with pytest.raises(NotADirectoryError):
        ingest.ingest_folder(tmp_path / "nope", source="x")
