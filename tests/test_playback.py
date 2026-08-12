"""판정 결과를 영상으로 되돌려 재생하는 경로 (V7).

여기서 지키려는 것은 «영상이 만들어졌는가»가 아니라 **영상이 거짓말을 하지 않는가**이다.
박스가 뜨는 순간과 «결함» 판정이 나는 순간이 어긋나거나, 열지도 색이 프레임마다 다른
기준으로 칠해지면, 보는 사람은 화면을 믿고 잘못된 결론을 내린다. 숫자를 보고 틀리는 것보다
그림을 보고 틀리는 쪽이 훨씬 알아채기 어렵다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vision_ai import config, playback, viz


# --- 열지도를 어디서 자를 것인가 ---------------------------------------------

def _map(peak: float, background: float = 1.0, size: int = 32) -> np.ndarray:
    grid = np.full((size, size), background, dtype=np.float32)
    grid[10:20, 10:20] = peak
    return grid


def test_the_cut_is_the_decision_threshold():
    """기본은 판정 기준선 그대로다 — 박스와 판정이 같은 선에서 갈려야 한다."""
    assert playback.cut_level(_map(5.0), 3.0, defect=True) == 3.0


def test_a_normal_frame_is_cut_at_the_threshold_too():
    """정상 판정 프레임을 다른 기준으로 자르면 «정상인데 박스가 있는» 화면이 나온다."""
    assert playback.cut_level(_map(2.0), 3.0, defect=False) == 3.0


def test_a_defect_frame_always_gets_something_to_look_at():
    """평균·백분위로 집계하면 «기준선은 넘겼는데 넘긴 화소는 없는» 상태가 가능하다.

    그때 기준선으로 자르면 박스가 하나도 없는 결함 프레임이 나온다. 그 화면은 아무것도
    알려주지 않는다 — 어디를 보고 그렇게 판정했는지는 보여 줘야 한다.
    """
    quiet = _map(2.0)                       # 최댓값 2.0 < 기준선 3.0
    level = playback.cut_level(quiet, 3.0, defect=True)
    assert level < 3.0
    assert playback.boxes_from_score_map(quiet, level, shape=(64, 64)), "박스가 하나는 나와야 한다"


# --- 열지도 → 박스 -----------------------------------------------------------

def test_a_hot_blob_becomes_one_box():
    boxes = playback.boxes_from_score_map(_map(9.0), 5.0, shape=(64, 64))
    assert len(boxes) == 1


def test_two_separate_blobs_become_two_boxes():
    """떨어진 두 곳을 한 박스로 묶으면 그 사이 배경까지 결함이라고 말하는 것이 된다."""
    grid = np.full((32, 32), 1.0, dtype=np.float32)
    grid[4:9, 4:9] = 9.0
    grid[22:27, 22:27] = 9.0
    assert len(playback.boxes_from_score_map(grid, 5.0, shape=(128, 128))) == 2


def test_the_box_lands_where_the_heat_is():
    """열지도는 모델 크기(격자)라 프레임 크기와 다르다. 좌표 환산이 틀리면 엉뚱한 곳에 그린다."""
    grid = np.full((32, 32), 1.0, dtype=np.float32)
    grid[16:24, 16:24] = 9.0                     # 오른쪽 아래 사분면
    (x, y, w, h), = playback.boxes_from_score_map(grid, 5.0, shape=(320, 320))
    assert x > 120 and y > 120, "왼쪽 위에 그렸다 — 좌표 환산이 뒤집혔다"
    assert x + w <= 320 and y + h <= 320


def test_specks_are_not_boxes():
    """열지도를 늘리면 잡음 한두 화소도 «덩어리»가 된다. 그것까지 그리면 화면이 못 쓰게 된다."""
    grid = np.full((256, 256), 1.0, dtype=np.float32)
    grid[128, 128] = 9.0
    assert playback.boxes_from_score_map(grid, 5.0, shape=(256, 256)) == []


def test_a_frame_that_is_hot_everywhere_does_not_get_two_hundred_boxes():
    """전면이 뜨거우면 답은 «전부»이지 박스 200개가 아니다."""
    rng = np.random.default_rng(0)
    grid = rng.uniform(6.0, 9.0, size=(64, 64)).astype(np.float32)
    grid[::2, ::2] = 1.0                          # 잘게 쪼개진 덩어리 수백 개
    assert len(playback.boxes_from_score_map(grid, 5.0, shape=(256, 256))) <= playback.MAX_BOXES


def test_nothing_over_the_line_means_no_boxes():
    assert playback.boxes_from_score_map(_map(2.0), 5.0, shape=(64, 64)) == []


def test_an_empty_map_is_not_an_error():
    assert playback.boxes_from_score_map(np.empty((0, 0)), 1.0, shape=(64, 64)) == []


# --- 열지도 색 ---------------------------------------------------------------

def test_the_colour_scale_is_anchored_to_the_threshold_not_the_frame():
    """**프레임마다 다시 정규화하면 안 된다.**

    정상 프레임의 미세한 얼룩이 결함과 똑같이 새빨갛게 나와서, 재생하는 내내 화면이 요동친다.
    같은 기준선을 쓰면 «조용한 프레임»과 «뜨거운 프레임»의 그림이 달라야 한다.
    """
    base = np.full((40, 40, 3), 120, dtype=np.uint8)
    quiet = playback.heat_overlay(base, _map(1.2, background=1.0), 5.0)
    hot = playback.heat_overlay(base, _map(9.0, background=1.0), 5.0)
    assert not np.array_equal(quiet, hot), "두 프레임이 같은 색으로 칠해졌다"


def test_cold_areas_keep_the_original_picture():
    """바닥까지 색으로 덮으면 왼쪽 원본과 대 보며 «저기가 뭐길래»를 따질 수 없다."""
    base = np.full((40, 40, 3), 120, dtype=np.uint8)
    painted = playback.heat_overlay(base, np.zeros((32, 32), dtype=np.float32), 5.0)
    assert np.abs(painted.astype(int) - base.astype(int)).max() <= 2


def test_the_overlay_keeps_the_frame_size():
    """열지도는 256×256이고 프레임은 아니다. 크기가 바뀌면 나란히 붙일 수 없다."""
    base = np.full((90, 160, 3), 120, dtype=np.uint8)
    assert playback.heat_overlay(base, _map(9.0), 5.0).shape == base.shape


# --- 나란히 붙이기 -----------------------------------------------------------

def test_two_panes_sit_side_by_side():
    left = np.zeros((50, 80, 3), dtype=np.uint8)
    right = np.zeros((50, 80, 3), dtype=np.uint8)
    joined = playback.side_by_side(left, right)
    assert joined.shape == (50, 80 * 2 + playback.DIVIDER_PX, 3)


def test_a_classifier_frame_is_not_split_in_two():
    """분류 모델은 결함의 **위치를 모른다.** 열지도 칸을 만들면 없는 정보를 지어내는 것이다."""
    frame = np.full((50, 80, 3), 100, dtype=np.uint8)
    judged = playback.Judged(0, 0.0, 0.9, 0.5, config.LABEL_DEFECT)
    assert playback.compose(frame, judged, None).shape[1] == 80


def test_an_anomaly_frame_is_split_in_two():
    frame = np.full((50, 80, 3), 100, dtype=np.uint8)
    judged = playback.Judged(0, 0.0, 9.0, 5.0, config.LABEL_DEFECT)
    assert playback.compose(frame, judged, _map(9.0)).shape[1] > 80 * 2


# --- 어디를 훑을 것인가 ------------------------------------------------------

def test_a_short_video_is_judged_frame_by_frame():
    assert playback.plan_stride(80, 150) == 1


def test_a_long_video_is_thinned_so_the_end_is_reached():
    """**앞에서부터 150장을 자르면 10분 영상의 앞 5초만 본 것이다.**

    조명이 바뀌거나 물건이 달라지는 뒷부분을 통째로 놓치는데, 화면에는 그 사실이 드러나지
    않는다 — 그냥 «결함 없음»으로 보인다.
    """
    stride = playback.plan_stride(18_000, 150)
    assert stride >= 120
    assert stride * 150 >= 18_000 * 0.99, "간격이 좁아 영상 끝에 못 닿는다"


def test_frame_count_is_never_zero_or_negative():
    assert playback.plan_stride(0, 150) == 1
    assert playback.plan_stride(500, 0) == 1


# --- 결함 구간과 대표 장면 ---------------------------------------------------

def _playback(pattern: str, threshold: float = 0.5) -> playback.Playback:
    """`pattern`의 `x`가 결함 프레임. 점수는 뒤로 갈수록 조금씩 세진다."""
    frames = [
        playback.Judged(
            index=i, seconds=i / 10.0,
            score=threshold + 0.1 + i * 0.01 if ch == "x" else threshold - 0.1,
            threshold=threshold,
            decision=config.LABEL_DEFECT if ch == "x" else config.LABEL_NORMAL,
        )
        for i, ch in enumerate(pattern)
    ]
    return playback.Playback(
        directory=Path("."), video=Path("judged.webm"), mime="video/webm", playable=True,
        frames=frames,
    )


def test_consecutive_defect_frames_are_one_event():
    """사람이 보기에 «한 군데서 잡음»인 것을 3건으로 세면 숫자가 사실과 어긋난다."""
    assert len(_playback("..xxx....").events()) == 1


def test_separated_defects_are_separate_events():
    assert len(_playback("..xx...xx...x").events()) == 3


def test_an_event_reports_the_seconds_it_covers():
    event, = _playback("..xxx....").events()
    assert event.span() == "0.2~0.4초"


def test_a_single_frame_event_reports_one_moment():
    event, = _playback("..x......").events()
    assert event.span() == "0.2초"


def test_highlights_do_not_repeat_the_same_moment():
    """이웃 프레임 4장을 «대표 4장»이라고 내놓으면 아무것도 대표하지 못한다."""
    picked = _playback("..xxxxxx....xx....x").highlights(limit=4)
    assert len({judged.index for judged in picked}) == len(picked)
    assert len(picked) == 3, "구간이 3곳인데 대표 장면이 그보다 많다"


def test_highlights_come_in_video_order():
    """세게 잡은 순서로 늘어놓으면 «영상의 어디쯤»인지 감이 잡히지 않는다."""
    picked = _playback("x....xxx....x").highlights()
    assert [j.index for j in picked] == sorted(j.index for j in picked)


def test_with_no_defects_the_closest_calls_are_shown():
    """«아무것도 안 잡혔습니다»는 다음에 무엇을 할지 알려주지 않는다."""
    quiet = _playback(".........")
    assert quiet.events() == []
    assert len(quiet.highlights(limit=3)) == 3


def test_the_summary_counts_frames_and_events():
    text = _playback("..xx...x..").summary()
    assert "10장" in text and "3장이 결함" in text and "2곳" in text


# --- 영상 한 편을 끝까지 만들기 ----------------------------------------------

class _FakeModel:
    """4단계가 쓰는 `serving.LoadedModel` 자리에 끼우는 최소 대역.

    실제 이상탐지 모델을 학습시키면 테스트가 몇 초씩 늘어나는데, 여기서 확인하려는 것은
    «모델이 잘 맞히는가»가 아니라 «판정이 영상·표·대표 장면에 일관되게 실렸는가»다.
    """

    kind = "anomaly"
    version = "vtest"
    threshold = 5.0

    def score_image(self, rgb) -> float:
        return 9.0 if rgb[:, :, 0].mean() > 100 else 1.0

    def score_map(self, rgb):
        grid = np.full((32, 32), 1.0, dtype=np.float32)
        if rgb[:, :, 0].mean() > 100:
            grid[8:20, 8:20] = 9.0
        return grid


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """앞 6장은 어둡고(정상), 가운데 4장은 밝은(결함) 20프레임짜리 영상."""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(20):
        value = 200 if 6 <= i < 10 else 30
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    return path


def test_a_playback_is_made_and_can_be_read_back(clip: Path, tmp_path: Path):
    made = playback.render(clip, _FakeModel(), limit=20, directory=tmp_path / "out")
    assert made.video.exists() and made.video.stat().st_size > 0
    assert len(made) == 20


def test_the_video_can_be_played_in_a_browser(clip: Path, tmp_path: Path):
    """만들어 놓고 브라우저가 재생을 못 하면 «눈으로 본다»는 목적 자체가 달성되지 않는다."""
    made = playback.render(clip, _FakeModel(), limit=20, directory=tmp_path / "out")
    assert made.playable, made.note()
    assert made.mime == "video/webm"


def test_the_original_video_is_left_alone(clip: Path, tmp_path: Path):
    """판정 기준을 바꿔 다시 돌리는 일이 잦다. 원본을 덮으면 되돌릴 방법이 없다."""
    before = clip.read_bytes()
    playback.render(clip, _FakeModel(), limit=20, directory=tmp_path / "out")
    assert clip.read_bytes() == before


def test_the_judgement_in_the_table_matches_the_video(clip: Path, tmp_path: Path):
    """밝은 4장만 결함이어야 한다 — 표와 화면이 다르면 무엇을 믿을지 알 수 없다."""
    made = playback.render(clip, _FakeModel(), limit=20, directory=tmp_path / "out")
    assert [j.index for j in made.defects] == [6, 7, 8, 9]


def test_defect_frames_carry_boxes_and_normal_frames_do_not(clip: Path, tmp_path: Path):
    made = playback.render(clip, _FakeModel(), limit=20, directory=tmp_path / "out")
    assert all(j.boxes for j in made.defects)
    assert all(not j.boxes for j in made.frames if not j.defect)


def test_a_still_is_written_for_the_representative_moment(clip: Path, tmp_path: Path):
    made = playback.render(clip, _FakeModel(), limit=20, directory=tmp_path / "out")
    assert made.stills and all(p.exists() and p.stat().st_size > 0 for p in made.stills)


def test_a_rerun_can_pick_the_result_back_up(clip: Path, tmp_path: Path):
    """Streamlit은 위젯을 건드릴 때마다 스크립트를 다시 돌린다. 메모리에만 있으면 사라진다."""
    made = playback.render(clip, _FakeModel(), limit=20, directory=tmp_path / "out")
    again = playback.load(tmp_path / "out")
    assert again is not None
    assert [j.decision for j in again.frames] == [j.decision for j in made.frames]
    assert again.mime == made.mime and again.version == made.version


def test_an_earlier_playback_is_found_again(clip: Path, monkeypatch, tmp_path: Path):
    """화면을 새로 열 때마다 몇 분짜리 작업을 다시 시키면 아무도 두 번 쓰지 않는다."""
    monkeypatch.setattr(config, "interim_dir", lambda: tmp_path / "interim")
    playback.render(clip, _FakeModel(), limit=10)
    found = playback.find(clip, "vtest")
    assert found is not None and len(found) == 10


def test_a_different_model_does_not_return_the_other_ones_playback(
    clip: Path, monkeypatch, tmp_path: Path
):
    """«어느 모델이 잡은 것인가»를 보려고 만드는 기능이다. 여기서 섞이면 결론이 뒤집힌다."""
    monkeypatch.setattr(config, "interim_dir", lambda: tmp_path / "interim")
    playback.render(clip, _FakeModel(), limit=10)
    assert playback.find(clip, "v999") is None


def test_a_different_video_does_not_return_this_ones_playback(
    clip: Path, monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(config, "interim_dir", lambda: tmp_path / "interim")
    playback.render(clip, _FakeModel(), limit=10)
    assert playback.find(tmp_path / "other.avi", "vtest") is None


def test_looking_for_a_playback_before_any_exists_is_not_an_error(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(config, "interim_dir", lambda: tmp_path / "nothing-here")
    assert playback.find(tmp_path / "clip.avi", "vtest") is None


def test_two_models_on_one_video_keep_separate_playbacks(clip: Path, monkeypatch, tmp_path: Path):
    """같은 영상을 다른 모델로 돌린 결과가 서로를 덮으면 비교 자체가 불가능해진다."""
    monkeypatch.setattr(config, "interim_dir", lambda: tmp_path / "interim")

    class _Other(_FakeModel):
        version = "vother"

    first = playback.render(clip, _FakeModel(), limit=10)
    second = playback.render(clip, _Other(), limit=10)
    assert first.directory != second.directory
    assert first.video.exists() and second.video.exists()


def test_a_half_written_folder_does_not_load(tmp_path: Path):
    """반쯤 살아난 판정본은 없는 것보다 나쁘다 — 어느 부분이 옛것인지 알 수 없다."""
    (tmp_path / "out").mkdir()
    assert playback.load(tmp_path / "out") is None


def test_the_previous_playback_is_cleared_before_the_next(clip: Path, tmp_path: Path):
    """지난 판정본의 대표 장면이 남아 섞이면 «어느 실행의 결과인지» 알 수 없게 된다."""
    target = tmp_path / "out"
    playback.render(clip, _FakeModel(), limit=20, directory=target)
    stale = target / playback.STILL_DIR / "99999999.jpg"
    stale.write_bytes(b"stale")
    made = playback.render(clip, _FakeModel(), limit=20, directory=target)
    assert not stale.exists()
    assert all(p.exists() for p in made.stills)


def test_a_long_video_is_sampled_not_truncated(clip: Path, tmp_path: Path):
    """20프레임짜리를 5장으로 볼 때, 앞 5장이 아니라 **영상 전체**에서 뽑아야 한다."""
    made = playback.render(clip, _FakeModel(), limit=5, directory=tmp_path / "out")
    assert made.frames[-1].index >= 12, "마지막 표본이 영상 끝 근처에 닿지 않았다"


def test_a_missing_video_says_so(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        playback.render(tmp_path / "nope.mp4", _FakeModel())


# --- 그림 위의 글자 ----------------------------------------------------------

def test_a_font_without_hangul_is_rejected():
    """**CJK라는 이름에 속으면 안 된다.** 일본어 고딕은 한자는 그리지만 한글은 네모로 찍는다.

    네모가 줄줄이 박힌 영상을 만들고 나서야 알게 되므로, 쓰기 전에 걸러야 한다.
    """
    japanese = Path("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf")
    if not japanese.exists():
        pytest.skip("이 환경에는 비교할 일본어 글꼴이 없다")
    assert viz.draws_hangul(str(japanese)) is False


def test_text_is_drawn_even_when_no_korean_font_exists(monkeypatch):
    """글꼴이 없다고 그림을 못 만들면 안 된다 — 영문으로라도 판정은 읽혀야 한다."""
    monkeypatch.setattr(viz, "korean_font_path", lambda: None)
    canvas = np.zeros((60, 300, 3), dtype=np.uint8)
    drawn = viz.put_text(canvas, "결함", (5, 5), ascii_fallback="DEFECT")
    assert drawn.sum() > 0, "아무것도 그려지지 않았다"


def test_drawing_text_does_not_change_the_picture_size():
    canvas = np.zeros((60, 300, 3), dtype=np.uint8)
    assert viz.put_text(canvas, "결함", (5, 5), ascii_fallback="DEFECT").shape == canvas.shape


def test_empty_text_leaves_the_picture_untouched():
    canvas = np.zeros((60, 300, 3), dtype=np.uint8)
    assert viz.put_text(canvas, "", (5, 5)) is canvas
