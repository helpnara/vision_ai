# CLAUDE.md

이 저장소에서 작업할 때 읽는 지침. 세션이 끊겨도 **여기부터 읽으면 다시 시작할 수 있게** 쓴다.

* 지금 상태·남은 백로그·검증용 프로젝트 → [`docs/HANDOFF.md`](docs/HANDOFF.md)
* 항목별 상세 사정과 결정 이력 → [`docs/todo.md`](docs/todo.md)
* 앱이 무엇을 하는가 → [`README.md`](README.md)

---

## 명령

```bash
# 의존성 (새 컨테이너는 비어 있다 — 테스트 전에 반드시)
pip install -r requirements.txt pytest ruff httpx

# 테스트 전체 (기준: 803 passed · 4 skipped = 807)
python -m pytest -q

# 한 파일 / 한 개
python -m pytest tests/test_segment_metrics.py -q
python -m pytest tests/test_api_labeling.py -q -k queue

# 앱 기동 (화면은 web/dist 에 빌드돼 있어 Node 없이 뜬다)
python serve.py                                   # 브라우저에서 열림 (8000)
python serve.py --no-browser --port 8000          # 원격/CI

# 기동 확인 (프록시 환경이므로 --noproxy 를 붙여야 로컬에 닿는다)
curl -s --noproxy '*' http://localhost:8000/api/health   # → {"ok":true,"project":"default"}

# 화면 코드를 고쳤을 때만 — 빌드 결과(web/dist)를 함께 커밋한다
cd web && npm install && npm run build            # tsc --noEmit + vite build

# 린트
ruff check .
```

`ruff check .`는 지금 **9건이 남아 있다**(미사용 import·f-string·모호한 변수명 `l`).
전부 기존 항목이고 기능에 영향이 없어 손대지 않았다. **새로 건드린 파일에서 이 수가 늘지
않는지만 보면 된다** — 특정 파일만 보려면 `ruff check <파일>`.

`pyproject.toml`이 `pythonpath = ["src"]`·`testpaths = ["tests"]`를 잡아 두므로
**pytest에 `PYTHONPATH=src`를 따로 줄 필요가 없다.** 스크립트는 pytest를 안 거치므로 필요하다:

```bash
PYTHONPATH=src python scripts/validate_visa.py
PYTHONPATH=src python scripts/measure_video.py <영상>
```

**건너뛰는 4건은 정상이다** — `tests/test_cnn_features.py`가 45MB 사전학습 가중치가 없으면
건너뛴다. 선택 기능이라 실패가 아니다.

---

## 코드 구조

UI와 코어 로직을 갈라 둔다. **`src/vision_ai/`는 화면을 모른다** (FastAPI도 React도 import하지
않는다) — 그래야 테스트가 화면 없이 돈다. 화면은 `server/`(JSON API)와 `web/`(React)에서만 조립한다.
2026-10 에 Streamlit을 떠나 이 구조로 옮겼다 — 규약은 [`docs/web-migration.md`](docs/web-migration.md).

```
serve.py            진입점. uvicorn으로 server.main:app 을 띄우고 브라우저를 연다
server/             FastAPI — 코어를 부르고 JSON으로 바꾸는 얇은 층. 판단 로직을 두지 않는다
  routers/          화면 하나 = 라우터 하나 (projects home ingest labeling modeling operations settings files jobs)
  jobs.py           긴 작업(학습·추출·추론·영상)을 스레드로 돌리고 진행률을 준다. 워커는 1개
  state.py          예전 st.session_state 자리 — 마지막 학습 결과·배치 추론 특징 (프로젝트 바꾸면 비운다)
  common.py         DataFrame → JSON (NaN→null). 응답에 pandas 값을 그대로 넣지 말 것
web/                React + TypeScript (Vite). pages/ 화면 · components/ 공용 조각 · api.ts hooks.ts
  dist/             빌드 결과 — git 추적. PC에서 Node 없이 python serve.py 만으로 돌게 하기 위해
src/vision_ai/      코어 로직 (화면 의존 없음)
tests/              pytest. conftest.py의 sandbox 픽스처가 경로를 tmp_path로 돌린다. test_api*.py 가 배선 테스트
scripts/            앱 밖에서 돌리는 것 (VisA 검증 · 영상 실측 · YOLO 학습)
data/ artifacts/    git 추적 제외 — 컨테이너가 죽으면 사라진다 (PC 로컬에서는 남는다)
```

### 4단계 파이프라인과 모듈

| 단계 | 라우터 / 화면 | 핵심 모듈 |
|---|---|---|
| 1 수집 | `routers/ingest.py` / `pages/Ingest.tsx` + `pages/ingest/` | `ingest` `storage` `datasets` `quality` `video` `framing`(V0 화각 점검) |
| 2 라벨링 | `routers/labeling.py` / `pages/Labeling.tsx` + `pages/labeling/` | `labeling` `boxes`(다중 박스) `detection`(YOLO 폴더 내보내기) |
| 3 모델 | `routers/modeling.py` / `pages/Modeling.tsx` + `pages/modeling/` | `features` `models` `evaluate` `experiments` `report` `claude_review` `cnn_features` |
| 4 운영 | `routers/operations.py` / `pages/Operations.tsx` + `pages/operations/` | `registry` `serving` `monitoring` `scenario` `segments`(V3) `compare`(V4) `playback`(V7) |
| 공통 | `routers/{projects,home,settings,files,jobs}.py` / `Layout.tsx` `Home.tsx` `Settings.tsx` | `config` `projects` `charts` `glossary` `guide` `settings` `viz` `quickstart` `feature_cache` `patch_cache` |

README의 «구조» 절에 모듈별 한 줄 설명이 있다.

### 꼭 알아야 할 설계 두 가지

**1. 경로는 컨테이너 둘에서 파생된다.** `config.DATA_HOME`·`config.ARTIFACT_HOME` 아래
프로젝트별로 갈린다. 테스트에서 경로를 격리할 때 개별 경로를 monkeypatch하지 말고
`conftest.py`의 `sandbox` 픽스처(또는 `homes`)를 쓴다 — **그 둘만 돌리면 나머지가 따라온다.**

**2. manifest를 직접 고치지 않는다.** 유효 라벨은 `labeling.resolve()`가 세 계층
(`manifest.csv` 수집 사실 / `labels.csv` 사람 판정 이력 / `splits.csv` 파생)을 겹쳐 계산한다.
**성능 측정의 정답으로 인정되는 것은 `label_source=human` · `verified=True`뿐이다** —
폴더 이름에서 추론한 라벨은 자기 채점이라 `monitoring.feedback_frame()`이 배제한다.

---

## 함정 규칙 — 새로 짜기 전에 읽을 것

아래는 **실제로 밟고 실측으로 뒤집힌 것**이다. 전부 테스트로 잠가 두었으니 고칠 때
테스트부터 읽으면 빠르다.

### 1. 지표를 새로 정의하면 «전 구간 알람 = 최악 모델»을 넣어 만점이 나오는지 본다

가장 크게 데인 곳이다. 헛알람을 «정답 구간과 겹치지 않은 예측 구간»으로 셌더니,
**전 프레임에 알람을 켠 모델이 구간 재현율 1.00 · 헛알람 0건**을 받았다. 전 구간에 알람을
켜면 예측 구간이 하나뿐이고 그것이 모든 정답 구간과 겹치기 때문이다. 화면은 «성능이
좋아졌다»고 말했고 실제로는 가장 나쁜 모델이었다.

> **구간·비율·건수로 된 지표를 새로 만들면, 반드시 «전부 결함» 입력을 넣어 본다.**
> 만점이 나오면 지표가 틀린 것이다. 보통 **분모나 «겹침» 정의**가 원인이다.
> 고칠 때는 **시간 비율**(정상인 시간 중 알람이 켜져 있던 비율) 같은,
> 구간 개수로 환원되지 않는 짝을 함께 낸다.

잠근 테스트 — 새 지표에도 같은 모양으로 하나 더 넣을 것:

```bash
python -m pytest tests/test_segment_metrics.py -q -k "shouts or flooding or always_on"
```

`test_a_model_that_shouts_defect_at_everything_does_not_score_perfectly`
· `test_an_always_on_alarm_is_called_out_in_the_summary`
· `test_a_model_that_shouts_paints_the_alarm_lane_orange`

**그리는 것도 같다.** 모델 줄에 «예측 구간»을 그대로 그리면 전 구간 알람이 화면상
«정답을 다 덮은 훌륭한 모델»로 보인다. 알람이 켜진 시간을 **정답 위/밖으로 갈라** 그려야
주황이 화면을 덮어 문제가 그림만으로 드러난다(`segments.bands`).

### 2. 차트 높이는 축·범례가 먼저 가져간다 — 실측으로 확인한다

Vega는 `height`를 **그림틀 전체 크기**로 보고 거기에 맞춰 줄인다. 축 눈금·축 제목·
범례가 먼저 자리를 가져가고 **남은 만큼만** 띠가 그려진다.

V6 타임라인을 150px로 두었더니 **정답 줄과 모델 줄이 한 줄로 포개졌다.** 어긋난 자리를
보라고 만든 화면인데 정작 두 줄이 겹친 것이다. H4 타임라인에서 이미 겪은 같은 함정이라
260px로 올렸다(`charts.BAND_HEIGHT`). 차트 명세는 서버(`charts.py`)가 만들고 화면은
`VegaChart`로 그리기만 한다 — 이런 실측값이 테스트로 잠기는 자리가 서버 쪽이기 때문이다.

> **줄 수를 늘리거나 범례를 붙이면 높이를 다시 실측한다.** 계산으로 맞히려 하지 말고
> 실제로 렌더해 눈으로 볼 것. 그리고 **줄 순서를 못 박는다** — 안 정하면 이름 순으로
> 정렬돼 «위가 정답»이라는 설명과 그림이 어긋난다(`charts.LANE_ORDER`).

### 3. 글꼴은 «파일이 있는가»가 아니라 «한글 글리프가 있는가»를 본다

이 환경의 일본어 고딕은 **한자는 그리는데 한글 자리에는 네모를 찍는다.** 파일 존재만
확인하면 네모가 줄줄이 박힌 영상을 다 만들고 나서야 알게 된다.

> **쓰기 전에 «없는 글자»와 같은 모양인지 대 본다.** `viz.draws_hangul(font_path)`가
> 한글 한 자와 없는 코드포인트를 각각 그려 픽셀이 같으면 `False`를 낸다.
> 한글 글꼴이 하나도 없으면 **영문으로 대체해 그린다**(`playback.pane_label`의 `ascii_fallback`).

### 4. 브라우저 재생은 mp4v가 안 된다 — WebM/VP8을 쓴다

이 OpenCV 빌드에는 H.264 인코더가 없고(라이선스 문제로 보통 빠져 있다), 있는 `mp4v`는
**크롬·파이어폭스가 재생하지 못한다.** 파일은 멀쩡히 만들어지므로 «만들었는데 안 보인다»로
나타나 원인을 찾기 어렵다.

> **`playback.CODECS`의 순서를 지킨다** — `VP80`/`.webm`(재생 가능)이 먼저고,
> `mp4v`/`.mp4`는 `playable=False`로 표시된 **최후 대비책**이다. 그쪽으로 떨어지면
> 화면에 «내려받아 보라»고 말해 준다.

### 5. Windows·한글 경로에서만 조용히 깨지는 것 — 리눅스 테스트로는 안 보인다

사용자는 **한국어 Windows PC**에서 쓴다(`바탕 화면`·한글 사용자 이름 폴더가 흔하다). 리눅스에서
멀쩡하던 «데모 한 바퀴 만들기»가 PC에서 «합성 샘플을 만들지 못했습니다»로 끝났다 —
Windows OpenCV는 한글 경로면 `cv2.imwrite`가 **예외 없이 False만 돌려주고** 파일을 안 쓴다.

> * OpenCV 파일 입출력(`imwrite/imread/VideoCapture/VideoWriter`)은 **`cvio`만** 거친다.
>   `tests/test_cvio.py`가 직접 호출을 찾아내 실패시킨다.
> * 열린 파일은 Windows에서 지우거나 덮어쓸 수 없다(WinError 32). 폴더 지우기는
>   `fsutil.remove_tree`(실패를 돌려준다), memmap은 **닫고 나서** `os.replace`.
> * 사용자가 친 이름이 폴더·파일 이름이 되면 `fsutil.safe_name`, 친 경로는 `common.typed_path`
>   (탐색기 «경로로 복사»가 큰따옴표를 붙인다), 밖에서 들어온 CSV는 UTF-8 또는 **cp949**.
> * 텍스트 파일은 언제나 `encoding="utf-8"`을 쓴다 — Windows 기본은 cp949다.
> * 새로 고친 Windows 문제는 `tests/test_windows_compat.py`에 «Windows처럼 행동하는 가짜»로 잠근다.

### 6. (덤) 열지도 색을 프레임마다 정규화하지 않는다

정상 프레임의 미세한 얼룩이 결함과 똑같이 새빨개져서 재생 내내 화면이 요동친다.
**판정 기준선을 색 눈금 한가운데에 고정**해야 «붉으면 기준선 위»가 영상 내내 같은 뜻이 된다
(`playback.cut_level`).

---

## 의존성 판올림 주의

`requirements.txt`는 **하한만** 지정한다(`fastapi>=0.115`). 새 컨테이너는 최신 판을 깔므로
검증된 조합과 벌어진다. **테스트가 갑자기 깨지면 코드 회귀부터 의심하지 말고 판을 비교할 것.**

실제로 겪은 것(Streamlit 시절): 1.60 → 1.64에서 테스트 도구의 상대경로 기준이 바뀌어 테스트
20개가 한꺼번에 죽었다. 교훈은 그대로다 — **테스트·스크립트에 넘기는 경로는 저장소 루트 기준
절대경로로 만든다** (`server/main.py`의 `ROOT`처럼).

프런트 의존성(`web/package.json`)은 `package-lock.json`으로 **고정**되어 있다. `npm install`은
lock을 따르므로 판이 벌어지지 않는다. 올리고 싶으면 일부러 올리고 빌드·스크린샷으로 확인한다.

---

## 작업 습관

* **브랜치**: `claude/stoic-albattani-qwcf61`에서 작업하고, **거기에 먼저 푸시한 뒤 배포
  브랜치로 fast-forward 한다.** 배포본이 보는 것은 `claude/vision-surface-defect-detection-j16xhj`
  다(README 배포표). 작업 브랜치에만 올리면 배포본에 반영되지 않는다.

  ```bash
  git push -u origin claude/stoic-albattani-qwcf61
  # 갈라진 것이 없는지 먼저 본다 — 참이어야 fast-forward다
  git merge-base --is-ancestor \
      origin/claude/vision-surface-defect-detection-j16xhj \
      origin/claude/stoic-albattani-qwcf61
  git push origin claude/stoic-albattani-qwcf61:claude/vision-surface-defect-detection-j16xhj
  git fetch origin && git branch -f claude/vision-surface-defect-detection-j16xhj \
      origin/claude/vision-surface-defect-detection-j16xhj   # 로컬 참조도 맞춘다
  ```

  **되돌려 쓰는 일이 없도록 `--is-ancestor`가 참일 때만 민다.** 거짓이면 배포 브랜치에
  작업 브랜치에 없는 커밋이 있다는 뜻이므로, 강제로 밀지 말고 무엇이 갈라졌는지 먼저 본다
  (`git log --oneline 작업브랜치..배포브랜치`). 배포 브랜치에서 직접 작업하지 않는다.
* **배포 zip**: 배포 브랜치까지 민 뒤 `python scripts/package_release.py --summary "한 줄"`로 만든다.
  이름은 **`느린 눈-YYYYMMDD-vNN.zip`**(사용자 지정 규칙 — 날짜는 KST, 번호는 날짜와 무관하게 계속
  오른다). zip 안 최상위 폴더는 언제나 `vision_ai/`(덮어쓰기 하려고). 스크립트가 `docs/releases.md`에
  한 줄 붙이므로 **그 변경을 커밋·푸시까지** 해야 다음 번호가 겹치지 않는다. 이름을 손으로 짓지 않는다.
* **커밋 단위**: 항목 하나 = 커밋 하나. 제목은 `V6: 놓친 곳을 타임라인에서 되짚기`처럼
  **항목 번호 + 사용자가 얻는 것**으로 쓴다. 기존 로그의 결을 따른다.
* **주석은 «왜»를 적는다.** 이 저장소는 «무엇을 하는가»보다 **«왜 그 값인가 · 무엇에 데였는가»**를
  코드 옆에 남기는 쪽이다(`ui.BAND_HEIGHT` 독스트링이 본보기). 실측으로 뒤집힌 것은
  숫자와 함께 남긴다.
* **끝낸 항목은 `docs/todo.md`의 체크박스를 닫고** «지금 열려 있는 항목» 표에서도 뺀다.
  그 표가 남은 일의 단일 출처다.
* **새 지표·새 차트·새 영상 출력을 넣었으면** 위 함정 규칙에 해당하는 회귀 테스트를 같이 넣는다.
* **화면(`web/src`)을 고쳤으면 `npm run build`로 `web/dist`를 갱신해 같은 커밋에 넣는다.** PC에서는
  dist만 실행되므로, 소스만 올리면 사용자 화면은 그대로다. 그리고 **Playwright로 스크린샷을 찍어
  눈으로 본다** (`docs/web-migration.md` §4) — 타입 검사는 레이아웃이 깨진 것을 잡지 못한다.
* **API를 새로 넣었으면 `tests/test_api_<화면>.py`에 배선 테스트를 넣는다.** 응답에 NaN이 섞이면
  JSON이 깨지므로 `common.jsonable/table/records`를 거쳐야 한다.
