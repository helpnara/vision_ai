import { useState } from "react";
import { useFetch } from "../hooks";
import { PageTitle } from "../components/Layout";
import { Caption, ErrorBox, Spinner, Tabs, useTab } from "../components/ui";
import { CatalogTab } from "./ingest/CatalogTab";
import { FolderTab } from "./ingest/FolderTab";
import { UploadTab } from "./ingest/UploadTab";
import { VideoTab } from "./ingest/VideoTab";
import { SyntheticTab } from "./ingest/SyntheticTab";
import { StatusTab } from "./ingest/StatusTab";
import type { IngestOptions } from "./ingest/types";

export function Ingest() {
  const { data: options, error, reload } = useFetch<IngestOptions>("/api/ingest/options");
  const [tab, setTab] = useTab(6);
  // 한 번 연 탭은 숨겨 두기만 한다 — Streamlit처럼 탭을 오가도 입력(고른 영상, 경로)이 남는다.
  const [visited, setVisited] = useState<Set<number>>(() => new Set([tab]));
  // 다른 탭에서 데이터가 바뀌면(등록·삭제) 현황 탭을 다시 그린다 — 예전 st.rerun()의 자리.
  const [dataVersion, setDataVersion] = useState(0);
  const change = (i: number) => { setVisited((v) => new Set(v).add(i)); setTab(i); };

  if (error) return <ErrorBox error={error} />;
  if (!options) return <Spinner />;

  // 탭이 6개인데 어디부터 눌러야 할지 표시가 없으면 초보자는 첫 탭부터 훑는다.
  // 데이터가 없을 때는 가장 빠른 길(합성 샘플)을 가리키고, 채워지면 완료로 바꾼다.
  const hasData = options.has_data;
  const mark = hasData ? "✅" : "👉";
  const tabs = [
    "🗂️ 오픈 데이터셋 카탈로그", "📁 로컬 폴더 임포트", "⬆️ 이미지 업로드", "🎞️ 영상에서 프레임 추출",
    `${mark} 🧪 합성 샘플 생성`, `${hasData ? "✅ " : ""}📊 수집 현황`,
  ];
  const onDataChanged = () => { setDataVersion((v) => v + 1); void reload(); };
  const panes = [
    <CatalogTab />,
    <FolderTab options={options} onDataChanged={onDataChanged} />,
    <UploadTab options={options} onDataChanged={onDataChanged} />,
    <VideoTab options={options} onDataChanged={onDataChanged} />,
    <SyntheticTab options={options} onDataChanged={onDataChanged} />,
    <StatusTab key={`status-${dataVersion}`} onDataChanged={onDataChanged} />,
  ];

  return (
    <>
      <PageTitle title="📥 1단계 · 데이터 수집 (입력)" caption="오픈 데이터셋과 직접 촬영 이미지를 하나의 manifest로 모은다. manifest는 이후 모든 단계의 입력이 된다." />
      <Tabs tabs={tabs} active={tab} onChange={change} />
      {!hasData ? <Caption>👉 표시된 탭이 가장 빠른 시작점입니다. 다운로드 없이 전 과정을 시험할 수 있습니다.</Caption> : null}
      {panes.map((pane, i) => (visited.has(i) || i === tab) ? (
        <div key={i} hidden={i !== tab} role="tabpanel">{pane}</div>
      ) : null)}
    </>
  );
}
