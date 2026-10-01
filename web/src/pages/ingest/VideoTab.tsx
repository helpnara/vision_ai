import { useEffect, useMemo, useRef, useState } from "react";
import { post } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { Gallery } from "../../components/Gallery";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Button, Caption, Checkbox, Cols, ErrorBox, Expander, Metric, NumberInput, Radio, Select, Slider, Spinner, TextInput, md } from "../../components/ui";
import type { IngestOptions } from "./types";
import { useDebounced, type Notice } from "./util";
import { VideoSource, type VideoEntry, type VideoSourceHandle } from "./VideoSource";

interface VideoInfo { path: string; fps: number; frame_count: number; width: number; height: number; duration_sec: number }
interface Plan { stride: number; expected_frames: number; per_second: number; frames_per_object: number; required_fps: number; feasible: boolean; reason: string }
interface ExtractResult {
  message: string; extract_message: string; video_id: string; added: number; duplicates: number;
  scanned: number; kept: number; drop_rate: number; forced: number; mostly_dropped: boolean; samples: string[];
}

/** 1단계 영상 선택 위젯의 상태 접두사. 공용 선택기(`VideoSource`)에 넘긴다. */
const VIDEO_PREFIX = "video";

/** 영상에서 학습용 프레임을 뽑는다 (설계: docs/video-frame-extraction-plan.md). */
export function VideoTab({ options, onDataChanged }: { options: IngestOptions; onDataChanged: () => void }) {
  const [path, setPath] = useState<string | null>(null);
  const [sampleNotice, setSampleNotice] = useState<Notice | null>(null);
  const [making, setMaking] = useState(false);
  const picker = useRef<VideoSourceHandle>(null);

  // 같은 파일을 다시 만들 때 재탐색(probe)이 다시 돌게 하는 세대 번호. 경로가 같으면 key가 안 바뀐다.
  const [generation, setGeneration] = useState(0);

  const makeSample = async () => {
    setMaking(true);
    // 이미 그 영상이 골라져 있으면 파일을 다시 쓰는 동안 probe·계획 요청이 반쯤 쓰인 파일을
    // 열어 400이 난다. 쓰는 동안은 아래 구역을 내려 두고, 다 만든 뒤 다시 고른다.
    setPath(null);
    try {
      const made = await post<VideoEntry>("/api/ingest/videos/sample");
      // 만든 영상이 곧바로 목록에서 골라진 상태가 되게 한다. 만들어 놓고 다시
      // 찾아 고르게 하면 한 단계를 더 시키는 것이다.
      picker.current?.pick(made.path);
      setGeneration((g) => g + 1);
      setSampleNotice({ kind: "success", icon: "✅", text: `만들었습니다: \`${made.path}\`` });
    } catch (e) { setSampleNotice({ kind: "error", text: (e as Error).message }); } finally { setMaking(false); }
  };

  return (
    <>
      {md("현장에서 데이터를 모으는 방식은 대체로 **영상 촬영**이다. 영상을 지정하면 일정 간격으로 프레임을 뽑아 manifest에 등록한다. 라벨은 2단계에서 붙인다.")}
      <Alert kind="info" icon="🔗">같은 영상에서 뽑은 프레임은 서로 매우 비슷합니다. 그래서 **영상 id를 그룹으로 달아** 2단계 분할이 통째로 같은 쪽(학습 또는 평가)에 넣을 수 있게 합니다. 섞이면 성능이 실제보다 높게 나옵니다.</Alert>

      <Expander title="영상이 없다면 — 시험용 영상 만들기">
        <Caption>물체가 3초마다 하나씩 지나가고 일부에 흠집이 있는 10초짜리 영상을 만듭니다. 다운로드 없이 조작을 익혀 볼 수 있습니다. **성능 근거로는 쓸 수 없습니다.**</Caption>
        <Button busy={making} onClick={() => void makeSample()}>🎬 시험용 영상 만들기</Button>
        {sampleNotice ? <Alert kind={sampleNotice.kind} icon={sampleNotice.icon}>{sampleNotice.text}</Alert> : null}
      </Expander>

      <VideoSource ref={picker} prefix={VIDEO_PREFIX} value={path} onChange={setPath} extensions={options.video_extensions} />
      {path ? <Probed key={`${path}#${generation}`} path={path} options={options} onDataChanged={onDataChanged} /> : null}
    </>
  );
}

function Probed({ path, options, onDataChanged }: { path: string; options: IngestOptions; onDataChanged: () => void }) {
  const { data: info, error } = useFetch<VideoInfo>(`/api/ingest/video/probe?path=${encodeURIComponent(path)}`);
  if (error) return <ErrorBox error={error} />;
  if (!info) return <Spinner />;
  return (
    <>
      <Cols n={4}>
        <Metric label="길이" value={`${info.duration_sec.toFixed(1)}초`} />
        <Metric label="fps" value={info.fps.toFixed(0)} />
        <Metric label="전체 프레임" value={info.frame_count.toLocaleString()} />
        <Metric label="해상도" value={`${info.width}×${info.height}`} />
      </Cols>
      <FramingCheck info={info} options={options} />
      <Extraction info={info} options={options} onDataChanged={onDataChanged} />
    </>
  );
}

// --- V0 화각 점검 --------------------------------------------------------------

interface FramingView {
  native_px: number; model_input_px: number; verdict: string; ok: boolean; hard: boolean; smallest_mm: number;
  safe_px: number; floor_px: number; advice: string;
  levers: { name: string; change: string; cost: string; reachable: boolean }[];
  hint: { fraction: number; source_fov_m: number; defect_mm: number } | null;
}

/**
 * 이 촬영으로 결함이 보이기는 하는가 (V0).
 *
 * **모델을 고르기 전에 답해야 하는 질문이다.** 결함이 3픽셀로 잡히면 어떤 모델을 써도
 * 안 되는데, 보통은 프레임을 다 뽑고 라벨링을 하고 학습을 돌린 뒤에야 그 사실이 드러난다.
 * 그래서 **추출 버튼 위에** 둔다 — 며칠 뒤에 알 일을 지금 알려주자는 것.
 */
function FramingCheck({ info, options }: { info: VideoInfo; options: IngestOptions }) {
  const [fovM, setFovM] = useState(1.0);
  const [defectMm, setDefectMm] = useState(10.0);
  const [model, setModel] = useState(options.model_inputs[0].name);
  const [tiles, setTiles] = useState(1);
  const [view, setView] = useState<FramingView | null>(null);
  const [error, setError] = useState<string | null>(null);
  // 입력을 바꾸면 즉시 다시 계산되어야 판단할 수 있다 — 숫자가 멈춘 뒤 서버에 묻는다.
  // 본문 객체는 값이 같을 때 같은 참조여야 한다 — 매 렌더마다 새 객체를 디바운스에 넣으면
  // 디바운스가 끝날 때마다 다시 렌더되어 250ms마다 서버를 부르는 고리가 된다 (실제로 그랬다).
  const body = useDebounced(useMemo(
    () => ({ sensor_px: info.width, fov_m: fovM, defect_mm: defectMm, model, tiles }),
    [info.width, fovM, defectMm, model, tiles],
  ), 250);
  useEffect(() => {
    let cancelled = false;
    post<FramingView>("/api/ingest/video/framing", body)
      .then((got) => { if (!cancelled) { setView(got); setError(null); } })
      .catch((e) => { if (!cancelled) setError((e as Error).message); });
    return () => { cancelled = true; };
  }, [body]);

  return (
    <Expander title="📏 이 촬영으로 결함이 보이는가 (추출 전에 확인)">
      <Caption>결함이 **모델 입력에서 몇 픽셀**이 되는지 계산합니다. 원본에서 100픽셀이어도 학습할 때 640으로 줄이면 그만큼 작아집니다 — 판정은 줄인 뒤 크기로 해야 합니다.</Caption>
      <Cols n={3}>
        <NumberInput label="카메라가 담는 폭 (m)" value={fovM} min={0.05} max={50} step={0.05} onChange={setFovM}
          help="카메라 한 대가 화면 가로에 담는 실제 폭. 컨베이어 한 줄이면 1m 안팎입니다." />
        <NumberInput label="잡아야 하는 가장 작은 결함 (mm)" value={defectMm} min={0.1} max={500} step={0.5} onChange={setDefectMm}
          help="짧은 변 기준입니다. 가장 작은 것을 넣으세요 — 그것이 기준을 정합니다." />
        <Select label="어디에 태울 것인가" value={model} options={options.model_inputs.map((m) => m.name)} onChange={setModel} />
      </Cols>
      <Slider label="타일 분할 (가로 등분 수)" value={tiles} min={1} max={8} step={1} onChange={setTiles}
        help="프레임을 잘라 조각마다 추론하면 같은 결함이 입력에서 커집니다. 소프트웨어만 고치면 되지만 추론 횟수가 등분 수의 제곱만큼 늘어납니다." />
      {error ? <ErrorBox error={error} /> : !view ? <Spinner /> : (
        <>
          <Cols n={3}>
            <Metric label="원본에서" value={`${view.native_px.toFixed(0)}px`} />
            <Metric label="모델 입력에서" value={`${view.model_input_px.toFixed(0)}px`} />
            <Metric label="판정" value={view.verdict} />
          </Cols>
          <Caption>{`이 촬영으로 잡을 수 있는 **가장 작은 결함은 약 ${view.smallest_mm.toFixed(0)}mm**입니다. (안정 기준 ${view.safe_px.toFixed(0)}px · 한계 ${view.floor_px.toFixed(0)}px)`}</Caption>
          {view.ok ? <Alert kind="success" icon="✅">{view.advice}</Alert>
            : view.hard ? <Alert kind="warning" icon="⚠️">{view.advice}</Alert>
              : <Alert kind="error" icon="🚫">{view.advice}</Alert>}
          {view.levers.map((lever) => lever.reachable ? (
            <Caption key={lever.name}>{`**${lever.name}** — ${lever.change} · ${lever.cost}`}</Caption>
          ) : (
            // 공용 inline()은 ~~ 안의 **를 풀지 않으므로 이 줄만 직접 그린다.
            <Caption key={lever.name}><s><strong>{lever.name}</strong> — {lever.change}</s> · <strong>원본에 그만한 정보가 없어 소용없습니다</strong> (확대는 없던 것을 만들지 못합니다)</Caption>
          ))}
          {/* 결함 크기를 모를 때, 이미 그려 둔 박스로 어림한다. 비율은 그 촬영을 어떻게 했느냐의
              결과이지 물리 상수가 아니므로, 그 촬영이 담던 폭을 함께 말한다. */}
          {view.hint ? (
            <Caption>{`참고 — 지금 이 프로젝트에 그려 둔 박스의 짧은 변 중앙값은 **화면 폭의 ${(view.hint.fraction * 100).toFixed(2)}%** 입니다. 그 이미지를 담던 폭이 ${view.hint.source_fov_m.toFixed(1)}m였다면 결함 실물은 약 **${view.hint.defect_mm.toFixed(0)}mm**입니다. 비율은 촬영 방식에 따라 달라지므로 실제로 한 번 재 보는 편이 훨씬 정확합니다.`}</Caption>
          ) : null}
        </>
      )}
    </Expander>
  );
}

// --- 추출 계획과 추출 ---------------------------------------------------------------

const MODE_RATE = "초당 장수로 지정";
const MODE_PROCESS = "공정 값으로 계산 (컨베이어)";
const MODES = [MODE_RATE, MODE_PROCESS] as const;
type Mode = (typeof MODES)[number];

/**
 * 간격을 정하는 방법을 고르게 한다.
 *
 * CCTV처럼 물체가 불규칙하게 지나가면 라인 속도로 계산할 수 없으므로 초당 장수 지정이
 * 기본이다. 컨베이어처럼 조건을 아는 경우에만 계산 모드가 의미가 있다.
 */
function Extraction({ info, options, onDataChanged }: { info: VideoInfo; options: IngestOptions; onDataChanged: () => void }) {
  const [mode, setMode] = useState<Mode>(MODE_RATE);
  const maxRate = Math.min(info.fps, 30);
  const [perSecond, setPerSecond] = useState(Math.min(2.0, maxRate));
  const [field, setField] = useState(0.3);
  const [speed, setSpeed] = useState(0.5);
  const [perObject, setPerObject] = useState(options.default_frames_per_object);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);

  // 위 FramingCheck과 같은 이유로 useMemo — 참조가 바뀌면 디바운스가 고리를 돈다.
  const body = useDebounced(useMemo(() => ({
    path: info.path, mode: mode === MODE_RATE ? "rate" : "process", per_second: perSecond,
    field_of_view_m: field, speed_mps: speed, frames_per_object: Math.round(perObject),
  }), [info.path, mode, perSecond, field, speed, perObject]), 250);
  useEffect(() => {
    let cancelled = false;
    post<Plan>("/api/ingest/video/plan", body)
      .then((got) => { if (!cancelled) { setPlan(got); setPlanError(null); } })
      .catch((e) => { if (!cancelled) { setPlan(null); setPlanError((e as Error).message); } });
    return () => { cancelled = true; };
  }, [body]);

  const [category, setCategory] = useState("video");
  const [dedupe, setDedupe] = useState(true);
  const [checkQuality, setCheckQuality] = useState(false);
  const job = useJob<ExtractResult>(() => onDataChanged());
  const result = job.job?.result ?? null;

  const start = () => {
    if (!plan) return;
    void job.start(() => post("/api/ingest/video/extract", {
      path: info.path, stride: plan.stride, category: category || "video", dedupe, check_quality: checkQuality,
    }));
  };

  return (
    <>
      <Radio<Mode> label="추출 간격 정하기" value={mode} options={MODES} onChange={setMode}
        help="CCTV는 물체 속도가 제각각이라 계산이 성립하지 않습니다. 초당 장수를 쓰세요." />
      {mode === MODE_RATE ? (
        <Slider label="초당 몇 장" value={perSecond} min={0.1} max={maxRate} step={0.1} format={(v) => v.toFixed(1)} onChange={setPerSecond} />
      ) : (
        <Cols n={3}>
          <NumberInput label="카메라 시야 길이 (m)" value={field} min={0.01} max={100} step={0.01} onChange={setField} help="화면 안에 들어오는 컨베이어 길이" />
          <NumberInput label="라인 속도 (m/s)" value={speed} min={0.01} max={50} step={0.01} onChange={setSpeed} />
          <NumberInput label="물체당 확보할 장수" value={perObject} min={1} max={50} step={1} onChange={setPerObject} help="흔들림·가림에 대비한 여유까지 포함한 값" />
        </Cols>
      )}
      {planError ? <Alert kind="error">{planError}</Alert> : null}
      {!plan ? null : (
        <>
          <Caption>{plan.reason}</Caption>
          {!plan.feasible ? (
            <Alert kind="warning" icon="🎥">**이 촬영으로는 목표 장수를 채울 수 없습니다.** 추출 설정이 아니라 촬영 계획의 문제이므로, 프레임을 뽑기 전에 라인 속도나 카메라를 조정하는 편이 낫습니다.</Alert>
          ) : null}
          <Cols n={2}>
            <Metric label="예상 추출 장수" value={`${plan.expected_frames.toLocaleString()}장`} />
            <Metric label="초당 추출" value={`${plan.per_second.toFixed(1)}장`} />
          </Cols>
          <Cols n={3}>
            <TextInput label="카테고리" value={category} onChange={setCategory} />
            <Checkbox label="거의 같은 프레임 버리기" checked={dedupe} onChange={setDedupe}
              help="정지 구간에서는 같은 그림이 쏟아집니다. 균등 추출만으로는 거를 수 없습니다." />
            <Checkbox label="흐린 프레임 버리기" checked={checkQuality} onChange={setCheckQuality}
              help="영상 프레임은 정지 이미지보다 전반적으로 흐립니다. 정지 이미지용 기준을 그대로 걸면 과하게 버릴 수 있어 기본은 꺼 둡니다." />
          </Cols>
          <Button primary busy={job.running} onClick={start}>🎞️ 프레임 추출 후 등록</Button>
          <JobProgress job={job.job} error={job.error} />
          {result ? <ExtractOutcome result={result} /> : null}
        </>
      )}
    </>
  );
}

function ExtractOutcome({ result }: { result: ExtractResult }) {
  return (
    <>
      <Caption>{result.extract_message}</Caption>
      <Alert kind="success" icon="✅">{`${result.message} · 그룹 \`${result.video_id}\``}</Alert>
      {result.duplicates ? <Caption>{`이미 등록된 프레임 ${result.duplicates.toLocaleString()}건은 건너뛰었습니다.`}</Caption> : null}
      {/* 대부분이 버려졌으면 그 사실을 말한다 (MOSTLY_DROPPED=0.8, 서버가 판정).
          "N장 추출"만 찍고 넘어가면 40장을 기대했는데 3장이 나온 것을 **2단계에 가서야** 안다.
          그때는 이미 추출 설정을 잊은 뒤다. 다만 이것이 꼭 오류는 아니다 — CCTV처럼 빈 장면이
          대부분이면 많이 버려지는 게 정상이다. 그래서 막지 않고 판단 근거만 준다. */}
      {result.mostly_dropped ? (
        <>
          <Alert kind="warning" icon="🧹">{`후보 ${result.scanned.toLocaleString()}장 중 **${Math.round(result.drop_rate * 100)}%를 버려** ${result.kept.toLocaleString()}장만 남았습니다. 빈 장면이 대부분인 영상이면 정상입니다. 그게 아니라면 **거의 같은 프레임 버리기**를 끄고 다시 뽑아 보세요.`}</Alert>
          {result.forced ? <Caption>{`연속으로 너무 오래 버려서 ${result.forced.toLocaleString()}장을 강제로 남겼습니다 — 이 표시가 보이면 중복 판정이 이 영상에 잘 안 맞는다는 뜻입니다.`}</Caption> : null}
        </>
      ) : null}
      {result.samples.length ? (
        <>
          <div><strong>추출 표본</strong></div>
          <Gallery items={result.samples.map((p) => ({ path: p }))} />
        </>
      ) : null}
      <Alert kind="info" icon="➡️">다음 — **2단계 라벨링**에서 결함 구간과 위치를 지정합니다.</Alert>
    </>
  );
}
