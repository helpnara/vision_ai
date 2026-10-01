import { PageTitle } from "../components/Layout";
import { Alert } from "../components/ui";

export function Ingest() {
  return (
    <>
      <PageTitle title="📥 1단계 · 데이터 수집 (입력)" caption="이식 중입니다." />
      <Alert kind="info">이 화면은 아직 이식 중입니다.</Alert>
    </>
  );
}
