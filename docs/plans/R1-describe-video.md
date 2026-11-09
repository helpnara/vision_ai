# R1: 영상 목록 이름 짓기를 코어 한 곳으로 — 1단계·4단계가 같은 영상을 같은 이름으로 보인다

> 단계: 설계
> 설계 승인: (대기)
> 규약: [`../process.md`](../process.md)

---

## 1. 설계

### 목적
영상 선택 목록에 보이는 이름(`line1.mp4 · 3.2MB`)을 만드는 함수가 **두 라우터에 글자 그대로 복제**돼 있다.

| 위치 | 이름 | 비고 |
|---|---|---|
| `server/routers/ingest.py:196` | `_describe_video(path)` | 독스트링에 «예전 `ui.describe_video` — ui는 Streamlit을 부르므로 여기서 다시 쓴다» |
| `server/routers/operations.py:566` | `_describe_video(path)` | 바로 위 주석: «1단계 쪽 엔드포인트가 함께 만들어지는 중이라 여기서는 자기 라우터 아래에 둔다. 나중에 하나로 합친다» |

본문은 같다(2026-11-09 대조). 전환 때 화면을 병렬로 옮기며 «공유 파일은 고치지 않는다»는 규약을 지키느라
생긴 복제다. 한쪽만 고치면 1단계와 4단계에서 **같은 영상이 다른 이름으로 보이게 된다** — 예를 들어 크기
표기 규칙을 바꾸면서 한쪽을 빠뜨리는 식이다. 사용자가 지금 겪는 문제는 없고, **앞으로 생길 어긋남을 막는**
작업이다. R2(영상 선택기 하나로)의 바닥이기도 하다.

또 이 함수는 «화면 문구»처럼 보이지만 판단(크기를 어떤 단위로 적어야 구분이 되는가)이 들어 있다.
CLAUDE.md는 **서버에 판단 로직을 두지 않는다**고 했으므로 코어가 제자리다.

### 범위
* `src/vision_ai/video.py`에 `describe_video(path) -> str`을 둔다. 본문·독스트링(«왜 크기만 쓰는가 · 왜 KB로
  내려가는가»)은 지금 것을 그대로 옮긴다.
* 두 라우터의 `_describe_video`를 지우고 `video.describe_video`를 부른다.
* 이 함수를 직접 잠그는 단위 테스트를 새로 넣는다(지금은 API 테스트가 `"MB" in label or "KB" in label`로만 간접 확인).

### 하지 않는 것
* **표기 규칙은 바꾸지 않는다** — 순수한 자리 옮김이다. 화면에 보이는 문자열은 바이트 단위로 같아야 한다.
  (빈 파일이 `1KB`로 보이는 것처럼 다듬을 여지가 있어도 손대지 않는다. 바꾸려면 따로 항목을 연다.)
* **두 라우터의 `/videos`·`/videos/check`·`/videos/upload` 응답 모양 차이는 그대로 둔다** → R2.
  (1단계는 `name`·`scan_dir`를 주고 check 실패를 404/400으로, 4단계는 `extensions`를 주고 `{"ok": False}`로 —
  두 화면이 각자 그 모양에 기대고 있으므로 합치는 일은 화면까지 함께 고쳐야 한다.)
* 화면(`web/src`)은 건드리지 않는다. 그래서 `web/dist`도 바뀌지 않는다.

### 바뀌는 곳

| 층 | 파일 | 무엇이 바뀌나 |
|---|---|---|
| 코어 | `src/vision_ai/video.py` | `describe_video(path)` 추가 (`listed()` 옆, «목록» 함수들 사이) |
| 서버 | `server/routers/ingest.py` | `_describe_video` 삭제 · `_video_entry`가 `video.describe_video` 호출 |
| 서버 | `server/routers/operations.py` | `_describe_video` 삭제 · `videos()`가 `video.describe_video` 호출. 위 주석의 «나중에 하나로 합친다»를 «이름 짓기는 R1에서 합침, 엔드포인트는 R2»로 고친다 |
| 테스트 | `tests/test_video.py` | 단위 테스트 추가 (아래) |

API 요청·응답 모양: **변화 없음.**

### 함정 규칙 점검 (CLAUDE.md)
| # | 규칙 | 해당? | 어떻게 지키나 |
|---|---|---|---|
| 1 | 새 지표에 «전부 결함» 입력 | 아니오 | 지표 아님 |
| 2 | 차트 높이·줄 순서 | 아니오 | 차트 아님 |
| 3 | 한글 글리프 | 아니오 | 그림 아님 |
| 4 | WebM/VP8 | 아니오 | 영상을 쓰지 않음 |
| 5 | Windows·한글 경로 | **예(읽기만)** | `path.stat()`과 `path.name`만 쓴다 — OpenCV를 거치지 않으므로 `cvio` 대상이 아니다. 한글 파일 이름이 이름 그대로 나오는지 테스트에 넣는다 |
| 6 | 열지도 색 | 아니오 | — |

### 검증 계획
* 새 테스트 (`tests/test_video.py`):
  * `test_describe_video_uses_kb_below_a_megabyte` — 300KB 파일 → `"a.mp4 · 300KB"`. **MB로 적으면 작은 영상이 전부
    `0.0MB`가 되어 구분이 안 된다**는 독스트링의 이유를 잠근다
  * `test_describe_video_uses_mb_from_a_megabyte` — 정확히 1,048,576바이트 → `"· 1.0MB"`, 2.5MB → `"· 2.5MB"`
  * `test_describe_video_tells_same_name_apart_by_size` — 다른 폴더의 같은 이름 두 개가 다른 문자열이 된다
  * `test_describe_video_falls_back_to_name_when_missing` — 없는 파일 → 이름만 (목록을 그리는 사이 지워져도 죽지 않는다)
  * `test_describe_video_keeps_korean_names` — `라인1 영상.mp4` → 이름이 그대로 앞에 온다
  * `test_routers_do_not_carry_their_own_describe_video` — 두 라우터 모듈에 `_describe_video`가 **없다**. 복제가
    다시 생기는 것을 막는 잠금이다(`test_cvio.py`가 직접 호출을 찾아내는 것과 같은 방식)
* 기존 API 테스트(`test_api_ingest.py` 영상 목록 · `test_api_operations.py` 영상 목록)가 그대로 통과 — 응답 모양 무변화 확인
* `python scripts/verify.py` (화면을 안 고치므로 `--web` 없이). pytest 기준 809 → **815**(새 테스트 6개)
* 스크린샷: 필요 없음 (화면 코드·응답 무변화)
* 사용자 확인: 필요 없음 (보이는 것이 같다). 배포 zip도 만들지 않는다

### 개발 조각
한 커밋이면 된다 — `R1: 영상 목록 이름 짓기를 코어 한 곳으로`.

### 열린 질문
* 없음. (표기 규칙을 바꾸지 않는 순수 이동이므로 사용자 판단이 필요한 선택지가 없다.)

---

## 2. 변경 기록 (개발 중 설계에서 벗어난 것)

| 날짜 | 무엇을 | 왜 |
|---|---|---|

---

## 3. 검증 결과

| # | 확인 | 결과 |
|---|---|---|
| 1 | pytest | |
| 2 | ruff | |
| 3 | 앱 기동 | |
| 4 | 빌드 · dist 차이 | 해당 없음 (화면 무변화) |
| 5 | 스크린샷 | 해당 없음 |
| 6 | 함정 회귀 | |
| 7 | 사용자 확인 | 해당 없음 |
