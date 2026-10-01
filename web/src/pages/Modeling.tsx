import { PageTitle } from "../components/Layout";
import { Tabs, useTab } from "../components/ui";
import { AnomalyTab } from "./modeling/AnomalyTab";
import { BaselineTab } from "./modeling/BaselineTab";
import { ClaudeTab } from "./modeling/ClaudeTab";
import { DataTab } from "./modeling/DataTab";
import { ExperimentsTab } from "./modeling/ExperimentsTab";
import { ReportTab } from "./modeling/ReportTab";

const TABS = ["📦 학습 데이터", "🧮 베이스라인", "🔎 이상탐지", "🤖 Claude 2차 판정", "📈 평가 리포트", "🧪 실험 기록"];

/**
 * 3단계: 모델 개발 · 평가. 탭 구조는 Streamlit 화면과 같다. 마지막 학습 결과는 서버
 * (`server/state.py`)가 들고 있으므로 탭을 옮겼다 돌아와도 평가 리포트가 그대로 보인다.
 */
export function Modeling() {
  const [tab, setTab] = useTab(TABS.length);
  return (
    <>
      <PageTitle title="🧠 3단계 · 모델 개발 · 평가" caption="베이스라인 → 이상탐지 → Claude 2차 판정 순으로 쌓는다. 미탐(결함을 놓침)을 최우선 리스크로 보고 재현율 중심으로 평가한다." />
      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      {tab === 0 ? <DataTab /> : null}
      {tab === 1 ? <BaselineTab /> : null}
      {tab === 2 ? <AnomalyTab /> : null}
      {tab === 3 ? <ClaudeTab /> : null}
      {tab === 4 ? <ReportTab /> : null}
      {tab === 5 ? <ExperimentsTab /> : null}
    </>
  );
}
