# 웹 화면 이식 규약 (Streamlit → FastAPI + React)

이 문서는 `app_pages/*.py`(Streamlit)를 `server/routers/*.py`(API) + `web/src/pages/*.tsx`(React)로
옮길 때 따르는 규약이다. 화면 여섯 개가 같은 모양·같은 배선을 갖게 하는 것이 목적이다.

## 1. 구조

```
server/                 FastAPI. 코어를 부르고 JSON으로 바꾸는 얇은 층. 판단 로직을 두지 않는다
  main.py               앱 생성, 라우터 등록, web/dist 정적 서빙 (SPA fallback)
  jobs.py               백그라운드 잡 러너 (워커 1개) — 진행률이 있는 긴 작업은 전부 이것으로
  state.py              예전 st.session_state 자리 (3단계 학습 결과·4단계 배치 추론 등)
  common.py             jsonable / records / table / counts / bad_request / not_found
  routers/<page>.py     화면 하나 = 라우터 하나. prefix는 /api/<page>
web/src/
  api.ts                get/post/put/del, useJob용 JobState, imageUrl/pathUrl/videoUrl, download*
  hooks.ts              useFetch(url) · useJob(onDone)
  components/ui.tsx     Metric Alert Caption Card Cols Divider Button Tabs useTab Expander
                        TextInput NumberInput Slider Select Checkbox Radio DataTable Bars Json Spinner fmt md
  components/JobProgress.tsx   잡 진행률 막대 (note·남은 시간 포함)
  components/Gallery.tsx       4열 이미지 격자
  components/VegaChart.tsx     서버가 만든 Vega-Lite 명세를 그린다 (vision_ai.charts)
  components/RoiPicker.tsx     이미지 위 드래그 (원본 픽셀 좌표) + zoomTo/zoomNote
  components/TimelinePicker.tsx 타임라인 구간 드래그
  components/Layout.tsx        사이드바·프로젝트 선택·PageTitle·useProject
  pages/<Page>.tsx      화면 하나. 긴 화면은 pages/<page>/ 폴더에 탭별 파일로 나눠도 된다
src/vision_ai/charts.py Vega-Lite 명세 (segment_timeline, line_chart). Streamlit 의존 없음
```

**공유 파일(`server/common.py` `server/jobs.py` `server/state.py` `server/main.py` `web/src/api.ts`
`web/src/hooks.ts` `web/src/components/*`)은 이식 작업 중에 고치지 않는다.** 필요한 것이 있으면
자기 라우터·자기 페이지 파일 안에 만들고, 보고서에 «공유 파일에 넣으면 좋을 것»으로 적는다.

## 2. API 규약

* 경로: `/api/<page>/<동작>`. 읽기는 GET, 바꾸는 것은 POST/PUT/DELETE. 본문은 pydantic 모델.
* 응답은 **항상 JSON-안전**해야 한다 — DataFrame은 `common.table(df)`(표) 또는 `common.records(df)`,
  value_counts는 `common.counts(series)`, 그 밖의 dict는 `common.jsonable(...)`로 감싼다.
  NaN을 그대로 돌려주면 응답이 깨진다.
* 오류: 코어의 `ValueError`는 그대로 올려도 된다(400으로 변환됨). 그 밖에는
  `raise bad_request("문장")` / `raise not_found("문장")`. 문장은 화면에 그대로 보인다 —
  Streamlit 화면이 `st.error`로 보여주던 문장을 그대로 쓴다.
* **진행률이 있는 작업(학습·추출·추론·생성·재생)은 잡으로** 돌린다:
  ```python
  from ..jobs import runner
  @router.post("/extract")
  def start_extract(body: Body) -> dict:
      def run(job):
          result = ingest.ingest_video(..., progress=job.progress)   # (done, total) 모양
          return {"...": ...}            # JSON-안전한 값. 화면이 job.result로 받는다
      job = runner.submit("프레임 추출 중", run, note="왜 오래 걸리는지 한 줄")
      return {"job_id": job.id}
  ```
  콜백 모양이 `(index, total, name)`이면 `job.progress3`를 넘긴다. 임의 문구를 쓰려면
  `lambda d, t: job.progress(d, t, f"정상 학습 {d}/{t}")`.
  잡 하나가 두 단계로 되어 있으면(학습 50% + 평가 50%) 비율을 직접 환산해 넘긴다.
* 세션 상태 대체: `from .. import state` → `state.get(state.KEY_...)` / `state.put(...)`.
  새 키가 필요하면 자기 라우터 안에 상수로 두고 `state.put("p4_xxx", ...)`처럼 쓴다.
* 이미지: manifest 이미지는 화면이 `imageUrl(image_id)`로 직접 가져간다. **계산해서 만든 그림**
  (히트맵·박스 오버레이·추출 표본)은 자기 라우터에 GET 엔드포인트를 만들고
  `from .files import image_response` 로 `image_response(rgb_array, width=...)`를 돌려준다.
  프로젝트 폴더 안의 파일 경로는 `pathUrl(path)`로 가져갈 수 있다.
* 업로드: `files: list[UploadFile] = File(...)` + `Form(...)`. 화면은 `FormData`를 `post()`에 넘긴다.
* 내려받기: 작은 텍스트(CSV·JSON·MD)는 JSON 응답에 문자열로 넣고 화면이 `downloadText()`로 저장.
  바이너리(zip·영상)는 `FileResponse`/`Response(bytes, media_type=...)` 엔드포인트 + `downloadUrl()`.
* 활성 프로젝트는 서버 전역이다(`config.use_project`). 라우터가 신경 쓸 것 없다.

## 3. 화면 규약

* 페이지 구조는 **Streamlit 화면의 탭 구조를 그대로** 옮긴다. 탭 이름·순서·👉/✅ 표식 규칙 포함.
  `const [tab, setTab] = useTab(N)` + `<Tabs tabs={[...]} active={tab} onChange={setTab} />`.
* **한국어 문구는 원문 그대로** 가져온다 — 설명·경고·캡션·도움말(help) 전부. 이 앱의 가치 절반은
  «왜 이렇게 보는가»를 설명하는 그 문장들이다. `st.markdown`/`st.caption`의 `**굵게**` 와
  `` `코드` `` 는 `md(text)` / `<Caption>` / `<Alert>` 가 그대로 그린다.
* 대응표:
  | Streamlit | React |
  |---|---|
  | `st.metric(label, value, help=, delta=)` | `<Metric label value help delta deltaInverse />` |
  | `st.columns(n)` + metric | `<Cols n={n}>` |
  | `st.success/info/warning/error(text, icon=)` | `<Alert kind="success|info|warning|error" icon=>` |
  | `st.caption` | `<Caption>` |
  | `st.markdown` | `{md(text)}` |
  | `st.dataframe(df)` | 서버 `table(df)` → `<DataTable table={...} help={...} scroll limit />` |
  | `st.bar_chart(value_counts, horizontal=True)` | 서버 `counts(series)` → `<Bars items title />` |
  | `st.line_chart(df)` | 서버 `charts.line_chart(records, x=, ys=)` → `<VegaChart spec />` |
  | `st.vega_lite_chart(segment_timeline)` | 서버 `charts.segment_timeline(bands, duration=)` → `<VegaChart />` |
  | `st.image(array)` 여러 장 | `<Gallery items=[{imageId|path|src, caption}] />` |
  | `st.button` | `<Button primary busy onClick>` |
  | `st.progress` + 긴 작업 | `useJob` + `<JobProgress job error />` |
  | `st.expander` | `<Expander title open>` |
  | `st.selectbox/radio/slider/number_input/text_input/checkbox` | 같은 이름의 컴포넌트 |
  | `st.download_button` | `<Button onClick={() => downloadText(...)}>` |
  | `st.page_link(page)` | `<Link to="/ingest">` (react-router) |
  | `st.rerun()` 뒤에 다시 읽기 | `reload()` (useFetch) |
  | `st.form` + submit | 로컬 state 모아 두었다가 버튼 한 번에 POST |
* 데이터 읽기: `const { data, error, reload } = useFetch<T>("/api/<page>/...", [deps])`.
  오류는 `<ErrorBox error={error} />`, 로딩은 `<Spinner />`.
* 버튼으로 바꾸는 동작: `post(...)` → 성공 메시지를 로컬 state에 두고 `<Alert kind="success">` →
  `reload()`. Streamlit의 `st.rerun()`이 하던 일이다.
* 숫자 표기는 `fmt.int / fmt.num / fmt.pct`. «잴 수 없는 값은 0으로 적지 않는다»(`—`).
* 주석은 «왜»를 적는다. Streamlit 화면의 독스트링에 있던 «왜 이 자리에 두는가 · 무엇에 데였는가»는
  옮긴 자리에 **그대로 남긴다.**

## 4. 검증

* API 테스트: `tests/test_api_<page>.py`. `conftest.sandbox` 픽스처로 경로를 격리하고
  `fastapi.testclient.TestClient(server.main.app)`로 부른다. 잡은 `runner`가 스레드로 돌리므로
  테스트에서는 `/api/jobs/{id}`를 폴링해 `done`까지 기다린다 (아래 helper를 복사해 써도 된다):
  ```python
  def wait_job(client, job_id, timeout=120):
      import time
      for _ in range(timeout * 10):
          body = client.get(f"/api/jobs/{job_id}").json()
          if body["status"] in ("done", "failed"):
              return body
          time.sleep(0.1)
      raise AssertionError("잡이 끝나지 않았다")
  ```
  `server.main`을 import하면 `projects.bootstrap()`이 **import 시점의** `config.DATA_HOME`을
  본다. sandbox 픽스처 안에서 `from server.main import app`을 해도 app은 모듈 캐시에 남으므로,
  테스트에서는 요청 전에 `projects.bootstrap()`을 다시 부르거나 `config` 경로만 격리되면 된다
  (라우터는 요청 때마다 `config.*()`를 호출하므로 sandbox만으로 격리된다).
* 눈으로 확인: 자기 포트·자기 데이터 루트로 서버를 띄우고 Playwright로 스크린샷을 찍는다.
  ```bash
  VISION_AI_DATA_ROOT=/tmp/.../data VISION_AI_ARTIFACT_ROOT=/tmp/.../artifacts \
    python serve.py --no-browser --port 81xx
  curl -s --noproxy '*' -X POST http://127.0.0.1:81xx/api/home/quickstart   # 데모 데이터
  ```
  ```python
  from playwright.sync_api import sync_playwright
  with sync_playwright() as p:
      b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
      pg = b.new_page(viewport={"width": 1280, "height": 900})
      errors = []; pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
      pg.goto("http://127.0.0.1:81xx/ingest#tab=3", wait_until="networkidle")
      pg.screenshot(path="shot.png", full_page=True)
      print(errors)
  ```
  화면 코드를 고친 뒤에는 `cd web && npm run build` (타입 검사 포함). 빌드 결과는 `web/dist`.
* 린트: `ruff check server tests` 가 새 오류를 내지 않아야 한다. `npm run build`가 `tsc --noEmit`을
  먼저 돌리므로 타입 오류는 빌드가 잡는다.
