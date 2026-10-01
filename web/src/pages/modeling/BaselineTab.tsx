import { useEffect, useState } from "react";
import { post } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Bars, Button, Caption, Checkbox, Cols, ErrorBox, Expander, Select, Slider, Spinner, md } from "../../components/ui";
import { AdviceBanner, EmptyStateLink, NoImages, TrainOutcome, type Advice, type EmptyLink, type TrainSummary } from "./shared";

interface BaselinePayload {
  empty: boolean;
  term: string;
  link: EmptyLink;
  last: TrainSummary | null;
  advice?: Advice;
  ready?: boolean;
  problems?: string[];
  kinds?: { key: string; label: string }[];
  splits?: string[];
  target_recall?: number;
  show_categories?: boolean;
  counts?: Record<string, number>;
}

export function BaselineTab() {
  const { data, error, reload } = useFetch<BaselinePayload>("/api/modeling/baseline");
  const [kind, setKind] = useState("logreg");
  const [balanced, setBalanced] = useState(true);
  const [evalSplit, setEvalSplit] = useState("test");
  const [targetRecall, setTargetRecall] = useState(0.95);
  const [useCategories, setUseCategories] = useState(false);
  const train = useJob<TrainSummary>(() => void reload());

  // 목표 재현율의 기본값은 설정 화면의 값이다 — 3·4단계가 같은 기준을 봐야 한다.
  useEffect(() => { if (data?.target_recall !== undefined) setTargetRecall(data.target_recall); }, [data?.target_recall]);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;

  const head = (
    <>
      <Caption>{`**베이스라인** — ${data.term}`}</Caption>
      {md("고전 CV 특징 + 분류기로 **성능 하한선**을 만든다. 무거운 모델 없이 곧바로 돌아가므로, 이후 모델이 이보다 나은지 판단하는 기준이 된다.")}
    </>
  );
  if (data.empty) return <>{head}<NoImages link={data.link} /></>;
  if (data.advice?.blocked) return <>{head}<AdviceBanner advice={data.advice} /></>;   // 이 데이터로는 불가능하다 — 안내에서 이유와 대안을 이미 설명했다
  if (!data.ready) {
    return (
      <>
        {head}
        <AdviceBanner advice={data.advice!} />
        {data.problems?.map((p) => <Alert key={p} kind="error" icon="🚧">{p}</Alert>)}
        <EmptyStateLink link={data.link} />
      </>
    );
  }

  const counts = data.counts!;
  const nEval = counts[evalSplit] ?? 0;
  const summary = train.job?.result ?? data.last;

  return (
    <>
      {head}
      <AdviceBanner advice={data.advice!} />
      <Cols n={3}>
        <Select label="분류기" value={kind} options={data.kinds!.map((k) => k.key)} labels={(k) => data.kinds!.find((x) => x.key === k)?.label ?? k} onChange={setKind} />
        <Checkbox label="클래스 불균형 보정" checked={balanced} onChange={setBalanced} help="정상이 결함보다 훨씬 많을 때 결함 쪽에 가중치를 준다." />
        <Select label="평가 분할" value={evalSplit} options={data.splits!} onChange={setEvalSplit} help="임계값 탐색은 val에서 하고 최종 보고는 test로 하는 것이 정석이다." />
      </Cols>
      <Slider label="목표 재현율 (미탐 최소화 기준)" value={targetRecall} min={0.5} max={1} step={0.01} format={(v) => v.toFixed(2)} onChange={setTargetRecall} help="이 재현율을 만족하는 임계값 중 오탐이 가장 적은 값을 자동 선택한다." />
      {/*
        카테고리별 임계값 스위치는 **학습 버튼보다 위에 둔다.** Streamlit에서 결과가 나온 뒤에
        켜는 자리에 두었더니 못 쓰는 기능이 됐다 — 위젯을 건드릴 때마다 화면을 다시 그리는데
        학습은 버튼을 누른 그 실행에서만 도므로, 체크하는 순간 결과가 통째로 사라졌다.
        분류기·목표 재현율과 같은 줄에 두면 그런 일이 없다. 카테고리가 하나뿐이면 아예 보이지 않는다.
      */}
      {data.show_categories ? (
        <Checkbox label="카테고리별 임계값" checked={useCategories} onChange={setUseCategories}
          help="카테고리는 서로 다른 제품이라 정상 분포부터 다릅니다. 임계값 하나를 전체에 쓰면 가장 어려운 카테고리가 전체를 끌어내립니다. 켜면 이 값들이 모델과 함께 등록되어 4단계 배치 추론에서도 그대로 쓰이고, 학습에 없던 새 카테고리는 전체 기준으로 판정합니다." />
      ) : null}
      {/* 장수는 라벨 표만 세면 알 수 있다. 특징 추출은 학습을 누른 뒤로 미룬다 — 화면을 열기만 해도 전량 추출하면 4,584장 기준 90초를 기다려야 한다. */}
      <Caption>{`학습 ${counts.train.toLocaleString()}장 · 평가(${evalSplit}) ${nEval.toLocaleString()}장`}</Caption>
      {nEval === 0 ? <Alert kind="error" icon="🚧">{`\`${evalSplit}\` 분할에 이미지가 없습니다.`}</Alert> : (
        <>
          <Button primary busy={train.running} onClick={() => void train.start(() => post("/api/modeling/baseline/train", {
            kind, balanced, eval_split: evalSplit, target_recall: targetRecall, use_categories: useCategories,
          }))}>🧠 학습 후 평가</Button>
          <JobProgress job={train.job} error={train.error} />
        </>
      )}
      {summary ? (
        <TrainOutcome summary={summary}>
          {summary.importance && summary.importance.length ? (
            <Expander title="특징 중요도 상위 15개"><Bars items={summary.importance} /></Expander>
          ) : null}
          <Alert kind="info" icon="📈">자세한 임계값 조정과 오탐/미탐 샘플은 **평가 리포트** 탭에서 확인합니다.</Alert>
        </TrainOutcome>
      ) : null}
    </>
  );
}
