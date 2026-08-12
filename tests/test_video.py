"""영상 프레임 추출 테스트.

이 기능의 핵심 질문은 "몇 장을 뽑는가"이고, 그 답이 틀리면 두 방향으로 손해가 난다.
적게 뽑으면 결함이 한 장도 안 찍히고, 많이 뽑으면 라벨링 비용만 늘고 정보량은 그대로다.
그래서 **간격 계산이 맞는지**와 **"이 촬영으로는 안 된다"를 말해 주는지**를 함께 본다.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_ai import video


@pytest.fixture
def clip(tmp_path):
    """물체 하나가 좌에서 우로 지나가는 짧은 영상."""
    import cv2

    path = tmp_path / "line.mp4"
    fps, frames, size = 30, 90, (320, 240)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), 60, np.uint8)
        x = int((index / frames) * (size[0] - 40))
        cv2.rectangle(frame, (x, 90), (x + 40, 150), (200, 200, 200), -1)
        writer.write(cv2.add(frame, rng.integers(0, 10, frame.shape, dtype=np.uint8)))
    writer.release()
    return path


@pytest.fixture
def still(tmp_path):
    """거의 움직이지 않는 영상 — 정지한 컨베이어나 빈 CCTV 장면."""
    import cv2

    path = tmp_path / "still.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    frame = np.full((120, 160, 3), 90, np.uint8)
    for _ in range(60):
        writer.write(frame)
    writer.release()
    return path


# --- 영상 정보 --------------------------------------------------------------

def test_probe_reads_the_metadata(clip):
    info = video.probe(clip)
    assert info.fps == pytest.approx(30, abs=0.5)
    assert info.frame_count == 90
    assert (info.width, info.height) == (320, 240)
    assert info.duration_sec == pytest.approx(3.0, abs=0.2)


def test_probe_refuses_a_file_that_is_not_a_video(tmp_path):
    broken = tmp_path / "not.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(OSError):
        video.probe(broken)


def test_video_extensions_are_recognised():
    assert video.is_video("a.MP4") and video.is_video("b.mov")
    assert not video.is_video("c.png")


# --- 몇 장을 뽑을 것인가 -----------------------------------------------------

def test_rate_plan_matches_the_requested_frames_per_second(clip):
    plan = video.plan_from_rate(video.probe(clip), per_second=5)
    assert plan.stride == 6                 # 30fps / 5장
    assert plan.expected_frames == 15       # 90프레임 / 6


def test_process_plan_follows_the_documented_arithmetic(clip):
    """설계안 §2.1의 예시와 같은 식이어야 한다."""
    plan = video.plan_from_process(
        video.probe(clip), field_of_view_m=0.30, speed_mps=0.5, frames_per_object=3
    )
    # 물체 통과 0.6초 × 30fps = 18프레임, 그중 3장 → 6프레임마다
    assert plan.stride == 6
    assert plan.frames_per_object == pytest.approx(18, abs=0.5)
    assert plan.feasible


def test_a_line_too_fast_for_the_camera_is_reported_as_impossible(clip):
    """이 계산의 값어치는 '이 촬영이 애초에 가능한가'를 미리 알려주는 데 있다."""
    plan = video.plan_from_process(
        video.probe(clip), field_of_view_m=0.30, speed_mps=5.0, frames_per_object=3
    )
    assert not plan.feasible
    assert plan.required_fps == pytest.approx(50, abs=1)
    assert "fps" in plan.reason


def test_plans_refuse_nonsense_input(clip):
    info = video.probe(clip)
    with pytest.raises(ValueError):
        video.plan_from_rate(info, per_second=0)
    with pytest.raises(ValueError):
        video.plan_from_process(info, field_of_view_m=0, speed_mps=1)
    with pytest.raises(ValueError):
        video.plan_from_process(info, field_of_view_m=1, speed_mps=1, frames_per_object=0)


# --- 추출 -------------------------------------------------------------------

def test_extract_saves_roughly_the_planned_number(clip, tmp_path):
    info = video.probe(clip)
    plan = video.plan_from_rate(info, per_second=5)
    result = video.extract(clip, stride=plan.stride, out_dir=tmp_path / "out", similarity=None)

    assert result.kept == plan.expected_frames
    assert all(path.exists() for path in result.saved)


def test_extracted_files_are_named_by_frame_number(clip, tmp_path):
    result = video.extract(clip, stride=30, out_dir=tmp_path / "out", similarity=None)
    names = [path.stem for path in result.saved]
    assert names == sorted(names), "프레임 순서대로 정렬되는 이름이어야 한다"


def test_near_identical_frames_are_dropped(still, tmp_path):
    """정지 구간에서는 같은 그림이 쏟아진다. 균등 추출만으로는 거를 수 없다."""
    kept_all = video.extract(still, stride=2, out_dir=tmp_path / "a", similarity=None)
    deduped = video.extract(still, stride=2, out_dir=tmp_path / "b", similarity=0.02)

    assert kept_all.kept > 1
    assert deduped.kept == 1, "움직임이 없으면 한 장이면 충분하다"
    assert deduped.dropped_similar == kept_all.kept - 1


def test_moving_footage_survives_duplicate_removal(clip, tmp_path):
    result = video.extract(clip, stride=6, out_dir=tmp_path / "out", similarity=0.005)
    assert result.kept > 5, "움직이는 영상까지 걸러내면 안 된다"


def test_limit_stops_early(clip, tmp_path):
    result = video.extract(clip, stride=1, out_dir=tmp_path / "out", similarity=None, limit=4)
    assert result.kept == 4


def test_progress_reaches_the_end(clip, tmp_path):
    seen: list[tuple[int, int]] = []
    video.extract(
        clip, stride=10, out_dir=tmp_path / "out", similarity=None,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen and seen[-1][0] == seen[-1][1]


def test_result_message_explains_what_was_dropped(still, tmp_path):
    """버린 것을 조용히 넘기면 '3,000장이라더니 왜 2,700장이지?'가 된다."""
    result = video.extract(still, stride=2, out_dir=tmp_path / "out", similarity=0.02)
    assert "제외" in result.as_message()


def test_extract_refuses_a_file_that_is_not_a_video(tmp_path):
    broken = tmp_path / "not.mp4"
    broken.write_bytes(b"nope")
    with pytest.raises(OSError):
        video.extract(broken, stride=1, out_dir=tmp_path / "out")


# --- 영상 id ----------------------------------------------------------------

def test_same_video_gets_the_same_id(clip):
    assert video.video_id(clip) == video.video_id(clip)


def test_different_videos_get_different_ids(clip, still):
    assert video.video_id(clip) != video.video_id(still)


def test_video_id_is_safe_as_a_folder_name(tmp_path, clip):
    import shutil

    awkward = tmp_path / "촬영 #1 (2공장).mp4"
    shutil.copy(clip, awkward)
    identifier = video.video_id(awkward)
    assert "/" not in identifier and " " not in identifier and "#" not in identifier


# --- 시험용 영상 --------------------------------------------------------------

def test_sample_video_is_readable(sandbox, tmp_path):
    """영상 파일이 없으면 이 기능을 시험조차 할 수 없다. 배포본은 올린 영상도 안 남는다."""
    from vision_ai import video as video_module

    made = video_module.make_sample(tmp_path / "sample.mp4", seconds=2)
    info = video_module.probe(made)
    assert info.usable
    assert info.frame_count == 60
    assert info.duration_sec == pytest.approx(2.0, abs=0.2)


def test_sample_video_actually_moves(sandbox, tmp_path):
    """정지 화면이면 중복 제거가 전부 걸러내 시험이 안 된다."""
    from vision_ai import video as video_module

    made = video_module.make_sample(tmp_path / "sample.mp4", seconds=3)
    result = video_module.extract(
        made, stride=5, out_dir=tmp_path / "out", similarity=video_module.DEFAULT_SIMILARITY
    )
    assert result.kept > 5


def test_sample_goes_into_the_project_by_default(sandbox):
    from vision_ai import config, video as video_module

    made = video_module.make_sample(seconds=1)
    assert config.interim_dir() in made.parents


# --- 고를 수 있는 영상 목록 --------------------------------------------------

def _fake_video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 1024)
    return path


def test_the_project_video_folder_is_scanned_without_being_asked(sandbox):
    """경로를 통째로 타이핑하게 하면 오타 하나로 실패하고, 무엇보다 어떤 영상이 이미
    올라와 있는지 화면에서 알 수가 없다."""
    _fake_video(video.video_dir() / "line1.mp4")
    assert [p.name for p in video.listed()] == ["line1.mp4"]


def test_files_that_are_not_videos_are_left_out(sandbox):
    _fake_video(video.video_dir() / "line1.mp4")
    _fake_video(video.video_dir() / "메모.txt")
    assert [p.name for p in video.listed()] == ["line1.mp4"]


def test_subfolders_are_scanned_too(sandbox):
    """CCTV 녹화본은 날짜 폴더로 쌓이는 것이 보통이다."""
    _fake_video(video.video_dir() / "2026-08-10" / "cam1.mp4")
    assert [p.name for p in video.listed()] == ["cam1.mp4"]


def test_extra_folders_can_be_scanned(sandbox, tmp_path):
    """로컬 실행에서는 영상이 프로젝트 밖(NAS 등)에 있는 것이 오히려 보통이다."""
    outside = _fake_video(tmp_path / "nas" / "cam9.mp4")
    assert outside in video.listed(tmp_path / "nas")


def test_the_same_video_is_not_listed_twice(sandbox):
    """영상 폴더를 '다른 폴더'로 또 적으면 목록에 두 번 나온다."""
    _fake_video(video.video_dir() / "line1.mp4")
    assert len(video.listed(video.video_dir())) == 1


def test_a_missing_folder_is_not_an_error(sandbox):
    """오타 난 폴더 때문에 화면이 죽으면 안 된다 — 목록만 비면 된다."""
    assert video.listed("/없는/폴더") == []


def test_the_listing_stops_at_the_limit(sandbox):
    """녹화본이 수천 개인 폴더를 통째로 selectbox에 넣으면 화면이 멈춘다."""
    for index in range(12):
        _fake_video(video.video_dir() / f"cam{index:02d}.mp4")
    assert len(video.listed(limit=5)) == 5


def test_the_video_folder_is_inside_the_project(sandbox):
    """프로젝트를 나눈 이유가 데이터를 섞지 않기 위해서다. 영상도 같다."""
    from vision_ai import config

    assert config.data_root() in video.video_dir().parents


# --- 프레임 되짚기 (H4 구간 라벨링의 토대) -----------------------------------

def test_frame_number_is_recoverable_from_the_file_name():
    """이 값이 있어야 타임라인 위에 프레임을 늘어놓고 구간으로 라벨할 수 있다."""
    assert video.frame_index("00000123.jpg") == 123
    assert video.frame_index("/a/b/00000000.jpg") == 0


def test_non_frame_files_report_no_number():
    assert video.frame_index("photo.png") is None
    assert video.frame_index("00001-copy.jpg") is None


def test_seconds_follow_the_frame_rate():
    assert video.frame_seconds("00000090.jpg", 30) == pytest.approx(3.0)
    assert video.frame_seconds("00000090.jpg", 0) is None


def test_extracted_frames_map_back_to_their_position(clip, tmp_path):
    """추출이 번호로 저장하므로 되짚기가 성립한다 — 둘이 어긋나면 구간 라벨이 엉킨다."""
    result = video.extract(clip, stride=10, out_dir=tmp_path / "out", similarity=None)
    indexes = [video.frame_index(path) for path in result.saved]
    assert indexes == sorted(indexes)
    assert all(index % 10 == 0 for index in indexes)
