import { PageTitle } from "../components/Layout";
import { Tabs, useTab } from "../components/ui";
import { ScenarioTab } from "./operations/ScenarioTab";
import { RegistryTab } from "./operations/RegistryTab";
import { InferenceTab } from "./operations/InferenceTab";
import { PlaybackTab } from "./operations/PlaybackTab";
import { CompareTab } from "./operations/CompareTab";
import { DriftTab } from "./operations/DriftTab";
import { PerformanceTab } from "./operations/PerformanceTab";
import { RetrainingTab } from "./operations/RetrainingTab";
import { TraceTab } from "./operations/TraceTab";

/** 4단계 화면. Streamlit `app_pages/p4_operations.py`의 9탭을 이름·순서 그대로 옮겼다. */
const TABS = [
  "🎬 운영 시나리오 시연", "📚 모델 레지스트리", "▶️ 배치 추론", "🎥 판정 영상",
  "🆚 영상 비교", "📉 드리프트 감시", "📈 성능 추이", "🔁 재학습 판단", "🔍 판정 이력",
];

export function Operations() {
  const [tab, setTab] = useTab(TABS.length);
  return (
    <>
      <PageTitle
        title="⚙️ 4단계 · 사후 운영관리 (MLOps)"
        caption="모델을 만드는 것보다 만든 뒤 성능이 떨어지는 것을 알아채는 일이 어렵다. 데이터 드리프트(정답 없이 감지)와 성능 드리프트(정답 필요)를 나눠 본다."
      />
      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      {/* 탭을 바꾸면 그 탭이 새로 읽는다 — Streamlit이 rerun마다 다시 계산하던 것과 같다 */}
      {tab === 0 ? <ScenarioTab /> : null}
      {tab === 1 ? <RegistryTab /> : null}
      {tab === 2 ? <InferenceTab /> : null}
      {tab === 3 ? <PlaybackTab /> : null}
      {tab === 4 ? <CompareTab /> : null}
      {tab === 5 ? <DriftTab /> : null}
      {tab === 6 ? <PerformanceTab /> : null}
      {tab === 7 ? <RetrainingTab /> : null}
      {tab === 8 ? <TraceTab /> : null}
    </>
  );
}
