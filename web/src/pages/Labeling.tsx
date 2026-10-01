import { useFetch } from "../hooks";
import { PageTitle, useProject } from "../components/Layout";
import { Caption, ErrorBox, Spinner, Tabs, useTab, md } from "../components/ui";
import { ReviewTab } from "./labeling/Review";
import { SegmentsTab } from "./labeling/Segments";
import { VerifyTab } from "./labeling/Verify";
import { MappingTab } from "./labeling/Mapping";
import { SplitTab } from "./labeling/Split";
import { StatusTab } from "./labeling/Status";
import type { Overview } from "./labeling/shared";

/**
 * 2단계: 라벨링. manifest는 그대로 두고 사람의 판정을 별도 이력으로 쌓는다.
 * 유효 라벨 = 폴더 추론 라벨 + 사람 판정 덮어쓰기 + 유형 정규화.
 */
export function Labeling() {
  const project = useProject();
  const { data, error, reload } = useFetch<Overview>("/api/labeling/overview", [project.version]);
  const [tab, setTab] = useTab(6);

  const title = <PageTitle title="🏷️ 2단계 · 라벨링" caption={md("manifest는 그대로 두고 사람의 판정을 별도 이력으로 쌓는다. 유효 라벨 = 폴더 추론 라벨 + 사람 판정 덮어쓰기 + 유형 정규화.")} />;
  if (error) return <>{title}<ErrorBox error={error} /></>;
  if (!data) return <>{title}<Spinner /></>;

  // 2단계에서 반드시 해야 하는 것은 **데이터 분할**이다. 나머지는 데이터에 따라 건너뛸 수
  // 있는데, 분할을 빼먹으면 3단계가 통째로 막힌다. 그래서 그 탭만 상태를 표시한다.
  const splitMark = data.split_done ? "✅" : "👉";
  const tabs = ["🔍 라벨 검수", "🎞️ 영상 구간 라벨링", "✅ 폴더 라벨 검증", "🔀 결함 유형 정규화", `${splitMark} ✂️ 데이터 분할`, "📊 라벨 현황"];

  return (
    <>
      {title}
      <Tabs tabs={tabs} active={tab} onChange={setTab} />
      {!data.split_done ? <Caption>👉 **데이터 분할**은 건너뛸 수 없습니다. 학습용과 평가용을 나눠 두지 않으면 3단계에서 학습을 시작할 수 없습니다.</Caption> : null}
      {tab === 0 ? <ReviewTab overview={data} /> : null}
      {tab === 1 ? <SegmentsTab overview={data} /> : null}
      {tab === 2 ? <VerifyTab overview={data} /> : null}
      {tab === 3 ? <MappingTab overview={data} /> : null}
      {tab === 4 ? <SplitTab onChanged={() => void reload()} /> : null}
      {tab === 5 ? <StatusTab /> : null}
    </>
  );
}
