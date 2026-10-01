import { useEffect, useState } from "react";
import { post } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { JobProgress } from "../../components/JobProgress";
import { Gallery } from "../../components/Gallery";
import { Alert, Button, Caption, Checkbox, Cols, Divider, ErrorBox, Expander, Metric, Radio, Select, Slider, Spinner, fmt, md } from "../../components/ui";
import { AdviceBanner, NoImages, TrainOutcome, type Advice, type EmptyLink, type TrainSummary } from "./shared";

interface AnomalyPayload {
  empty: boolean;
  term: string;
  link: EmptyLink;
  last: TrainSummary | null;
  heatmap: { available: boolean; defects: string[] };
  advice?: Advice;
  splits?: string[];
  backends?: { key: string; label: string }[];
  backend_cnn?: string;
  cnn?: { available: boolean; describe: string; size_mb: number };
  score_modes?: { key: string; label: string }[];
  patch_feature_dim?: number;
  target_recall?: number;
  show_categories?: boolean;
  counts?: { train_normal: number; eval: Record<string, { total: number; defect: number }> };
}

export function AnomalyTab() {
  const { data, error, reload } = useFetch<AnomalyPayload>("/api/modeling/anomaly");
  const [evalSplit, setEvalSplit] = useState("test");
  const [backend, setBackend] = useState("classic");
  const [perPosition, setPerPosition] = useState(true);
  const [scoreMode, setScoreMode] = useState("p99");
  const [targetRecall, setTargetRecall] = useState(0.95);
  const [useCategories, setUseCategories] = useState(false);
  const train = useJob<TrainSummary>(() => void reload());
  const download = useJob<{ describe: string }>(() => void reload());

  useEffect(() => { if (data?.target_recall !== undefined) setTargetRecall(data.target_recall); }, [data?.target_recall]);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;

  const head = (
    <>
      <Caption>{`**이상탐지** — ${data.term}`}</Caption>
      {md("**정상 이미지만** 학습해 정상 분포를 만들고, 이탈 정도를 결함 점수로 쓴다. 결함 샘플이 적은 상황(VisA의 기본 전제)에 맞고, 결함 **위치**까지 히트맵으로 낸다.")}
    </>
  );
  if (data.empty) return <>{head}<NoImages link={data.link} /></>;
  if (data.advice?.blocked) return <>{head}<AdviceBanner advice={data.advice} /></>;

  const counts = data.counts!;
  const evalRows = counts.eval[evalSplit] ?? { total: 0, defect: 0 };
  const dim = data.patch_feature_dim!;
  const cnnKey = data.backend_cnn!;
  const cnnReady = !!data.cnn?.available;
  // CNN을 골랐는데 모델 파일이 없으면 원문처럼 고전 CV로 돌린다 (서버도 같은 규칙).
  const effectiveBackend = backend === cnnKey && !cnnReady ? "classic" : backend;
  const summary = train.job?.result ?? data.last;
  const canTrain = counts.train_normal > 0 && evalRows.total > 0;

  return (
    <>
      {head}
      <AdviceBanner advice={data.advice!} />
      <Cols n={3}>
        <Select label="평가 분할" value={evalSplit} options={data.splits!} onChange={setEvalSplit} />
        <Checkbox label="위치별 분포 학습" checked={perPosition} onChange={setPerPosition}
          help={`격자 위치마다 별도 정상 분포를 학습한다. PCB처럼 정렬된 이미지에 유리하지만 정상 이미지가 특징 차원(${dim})보다 많아야 한다.`} />
        <Select label="이미지 점수 집계" value={scoreMode} options={data.score_modes!.map((m) => m.key)} labels={(k) => data.score_modes!.find((m) => m.key === k)?.label ?? k} onChange={setScoreMode}
          help="격자 점수들을 이미지 1장의 점수로 합치는 방식. 결함이 작으면 max 쪽이 민감하다." />
      </Cols>
      {/*
        특징 추출 방식. **기본은 고전 CV다.** 사전학습 CNN을 측정해 보니 이미지 판정은 조금
        나아지지만(평균 AUROC 0.830 → 0.846, AP 0.811 → 0.851) **위치 찾기는 크게 나빠진다**
        (적중률 0.194 → 0.013). OpenCV로는 마지막 8×8 특징맵만 쓸 수 있어 미세 결함을 짚기에
        너무 성기기 때문이다. 그래서 권장이 아니라 선택지로 둔다.
      */}
      <Expander title="특징 추출 방식 (기본: 고전 CV)">
        <Caption>**측정 결과**(VisA PCB 4종): 사전학습 CNN은 이미지 판정이 조금 나아지지만(평균 AUROC 0.830 → 0.846) **결함 위치 찾기는 크게 나빠집니다**(적중률 0.194 → 0.013). 위치 히트맵이 필요하면 고전 CV를 쓰세요.</Caption>
        <Radio label="방식" value={backend} options={data.backends!.map((b) => b.key)} labels={(k) => data.backends!.find((b) => b.key === k)?.label ?? k} onChange={setBackend} />
        {backend === cnnKey ? (
          <>
            <Caption>{data.cnn!.describe}</Caption>
            {cnnReady ? null : (
              <>
                <Alert kind="warning" icon="⬇️">{`이 방식은 사전학습 모델 파일(약 ${data.cnn!.size_mb}MB)이 필요합니다. 저장소에 넣기엔 커서 필요할 때 내려받습니다.`}</Alert>
                <Button busy={download.running} onClick={() => void download.start(() => post("/api/modeling/anomaly/download-cnn"))}>⬇️ 모델 내려받기</Button>
                <JobProgress job={download.job} error={download.error} />
                {download.job?.status === "done" ? <Alert kind="success">내려받았습니다.</Alert> : null}
              </>
            )}
          </>
        ) : null}
      </Expander>
      <Slider label="목표 재현율" value={targetRecall} min={0.5} max={1} step={0.01} format={(v) => v.toFixed(2)} onChange={setTargetRecall} />
      {/* 카테고리 스위치는 학습 버튼보다 위 — 이유는 BaselineTab 주석과 같다. */}
      {data.show_categories ? (
        <Checkbox label="카테고리별 임계값" checked={useCategories} onChange={setUseCategories}
          help="카테고리는 서로 다른 제품이라 정상 분포부터 다릅니다. 임계값 하나를 전체에 쓰면 가장 어려운 카테고리가 전체를 끌어내립니다. 켜면 이 값들이 모델과 함께 등록되어 4단계 배치 추론에서도 그대로 쓰이고, 학습에 없던 새 카테고리는 전체 기준으로 판정합니다." />
      ) : null}
      <Caption>{`정상 학습 ${counts.train_normal.toLocaleString()}장 · 평가(${evalSplit}) ${evalRows.total.toLocaleString()}장 (결함 ${evalRows.defect.toLocaleString()}장)`}</Caption>
      {!canTrain ? <Alert kind="error" icon="🚧">학습용 정상 이미지와 평가 분할이 모두 필요합니다.</Alert> : (
        <>
          {perPosition && counts.train_normal <= dim ? (
            <Alert kind="warning">{`정상 이미지가 ${counts.train_normal}장으로 특징 차원(${dim})보다 많지 않습니다. '위치별 분포 학습'을 끄거나 정상 이미지를 늘리세요.`}</Alert>
          ) : null}
          <Button primary busy={train.running} onClick={() => void train.start(() => post("/api/modeling/anomaly/train", {
            eval_split: evalSplit, backend: effectiveBackend, per_position: perPosition, score_mode: scoreMode,
            target_recall: targetRecall, use_categories: useCategories,
          }))}>🔎 이상탐지 학습 후 평가</Button>
          <JobProgress job={train.job} error={train.error} />
        </>
      )}
      {summary ? <TrainOutcome summary={summary} /> : null}
      {data.heatmap.available ? <HeatmapSection defects={data.heatmap.defects} /> : null}
    </>
  );
}

interface HeatmapInfo { image_id: string; localization: { hit: boolean; iou: number | null; pixel_auroc: number | null } | null; note: string | null }

/** 학습된 이상탐지 모델로 결함 위치 히트맵을 보여준다. 이미지 3장은 서버가 계산해 준다. */
function HeatmapSection({ defects }: { defects: string[] }) {
  const [picked, setPicked] = useState<string | null>(defects[0] ?? null);
  const { data, error } = useFetch<HeatmapInfo>(picked ? `/api/modeling/anomaly/heatmap/${encodeURIComponent(picked)}` : null, [picked]);
  return (
    <>
      <Divider />
      <h3>결함 위치 히트맵</h3>
      {defects.length === 0 ? <Caption>결함 이미지가 없습니다.</Caption> : (
        <>
          <Select label="이미지" value={picked} options={defects} onChange={setPicked} />
          {picked ? (
            <Gallery cols={3} items={["input", "heatmap", "overlay"].map((kind, i) => ({
              src: `/api/modeling/anomaly/heatmap/${encodeURIComponent(picked)}/${kind}`,
              caption: ["입력", "결함 점수 히트맵", "상위 영역 오버레이"][i],
            }))} />
          ) : null}
          <ErrorBox error={error} />
          {data?.note ? <Caption>{data.note}</Caption> : null}
          {data?.localization ? (
            <Cols n={3}>
              <Metric label="최고점 명중" value={data.localization.hit ? "예" : "아니오"} />
              <Metric label="픽셀 AUROC" value={fmt.num(data.localization.pixel_auroc)} />
              <Metric label="IoU@p99" value={fmt.num(data.localization.iou)} />
            </Cols>
          ) : null}
        </>
      )}
    </>
  );
}
