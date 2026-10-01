import { useState } from "react";
import type { Table } from "../../api";
import { useFetch } from "../../hooks";
import { Alert, Caption, Cols, DataTable, ErrorBox, Expander, Select, Spinner, md } from "../../components/ui";

interface Dataset {
  key: string; name: string; summary: string; url: string; license: string; commercial_use: string;
  layout: string; layout_note: string; categories: string[]; fit_stars: string; everyday_note: string;
  download_note: string; license_note: string; tags: string[]; is_default: boolean;
}
interface Catalog { default: { name: string; license: string }; table: Table; help: Record<string, string>; datasets: Dataset[] }

export function CatalogTab() {
  const { data, error } = useFetch<Catalog>("/api/ingest/catalog");
  const [selected, setSelected] = useState<string | null>(null);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  const dataset = data.datasets.find((d) => d.name === (selected ?? data.datasets[0].name)) ?? data.datasets[0];

  return (
    <>
      {md("사내 데이터를 쓰지 않으므로 공개 데이터셋이 학습 데이터의 출발점이다. 아래는 표면 결함 분야에서 널리 쓰이는 데이터셋을 정리한 것이다.")}
      <Alert kind="success" icon="⭐">{`**기본 예시 데이터셋: ${data.default.name}** — 라이선스가 \`${data.default.license}\`로 상업적 이용이 가능해, 이후 회사 업무로 연장할 때 데이터셋을 갈아치우지 않아도 된다.`}</Alert>
      <Alert kind="warning" icon="⚠️">라이선스·URL은 정리 시점 기준 정보다. 특히 **비상업(NC)** 조건이 붙은 데이터셋이 많으므로, 사용 전 원본 배포 페이지에서 조건을 직접 확인할 것.</Alert>

      {/* 열이 7개라 좁은 화면에서는 폭에 맞추려다 글자가 뭉개진다. DataTable은 가로 스크롤이 생기므로
          찌그러지지 않는다. ⭐ 표시는 별도 열이었으나 이름 앞에 붙여 열을 하나 줄였다. */}
      <DataTable table={data.table} help={data.help} />

      <h4>상세 정보</h4>
      <Select label="데이터셋 선택" value={dataset.name} options={data.datasets.map((d) => d.name)} onChange={setSelected} />
      {md(`**${dataset.name}** — ${dataset.summary}`)}
      <Cols n={2}>
        <div>
          {md(`- 배포 페이지: ${dataset.url}\n- 라이선스: \`${dataset.license}\` (상업적 이용: ${dataset.commercial_use})\n- 일상 물건 적합도: ${dataset.fit_stars}`)}
          {dataset.tags.length ? <Caption>{"태그: " + dataset.tags.join(", ")}</Caption> : null}
        </div>
        <div>
          {md(`- 폴더 구조(\`${dataset.layout}\`): \`${dataset.layout_note}\`\n- 다운로드: ${dataset.download_note}`)}
        </div>
      </Cols>
      <Alert kind="info" icon="🏠">{`일상 물건 관점: ${dataset.everyday_note}`}</Alert>
      <Caption>{`⚠️ ${dataset.license_note}`}</Caption>
      <Expander title="카테고리 목록">{dataset.categories.length ? dataset.categories.join(", ") : "정보 없음"}</Expander>
      <Alert kind="success" icon="➡️">다운로드·압축 해제까지 마쳤다면 **로컬 폴더 임포트** 탭에서 폴더 경로를 지정해 등록한다.</Alert>
    </>
  );
}
