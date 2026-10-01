import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { imageUrl, type Table } from "../../api";
import { useFetch } from "../../hooks";
import { Alert, Caption, Card, Cols, DataTable, Divider, ErrorBox, Select, Spinner, md } from "../../components/ui";

interface Payload { empty: boolean; candidates: string[] }
interface Trace {
  image_id: string; known: boolean; manifest: string | null; effective: string | null;
  events: Table | null; inferences: Table | null; reviews: { title: string; reason: string }[];
}

/** 이미지 하나가 왜 그렇게 판정됐는지 되짚는다. 1~4단계 산출물이 모두 image_id로 연결되어 있다. */
export function TraceTab() {
  const { data, error } = useFetch<Payload>("/api/operations/trace");
  const [picked, setPicked] = useState<string | null>(null);
  useEffect(() => { if (data && picked === null && data.candidates.length) setPicked(data.candidates[0]); }, [data, picked]);
  const trace = useFetch<Trace>(picked ? `/api/operations/trace/${encodeURIComponent(picked)}` : null, [picked]);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  const intro = md("이미지 하나가 **왜 그렇게 판정됐는지**를 되짚는다. 1~4단계 산출물이 모두 `image_id`로 연결되어 있어 한 화면에 모을 수 있다.");
  if (data.empty) return <>{intro}<Alert kind="info" icon="📥">수집된 이미지가 없습니다.</Alert><Link to="/ingest">➡️ 1단계 데이터 수집으로 이동</Link></>;
  if (!data.candidates.length) return <>{intro}<Alert kind="info" icon="🔍">조회할 이미지가 없습니다.</Alert></>;

  const t = trace.data;
  return (
    <>
      {intro}
      <Select label="이미지" value={picked} options={data.candidates} onChange={setPicked} help="추론 로그가 있으면 판정된 이미지만 보여준다." />
      {trace.error ? <ErrorBox error={trace.error} /> : null}
      {!t ? <Spinner /> : (
        <>
          <Cols n={2}>
            <div>{t.known ? <img src={imageUrl(t.image_id, 800)} alt={t.image_id} style={{ width: "100%", borderRadius: 6 }} /> : null}{t.known ? <Caption>{t.image_id}</Caption> : null}</div>
            <div>
              {t.manifest ? <>{md("**① 수집 (1단계)**")}<Caption>{t.manifest}</Caption></> : null}
              {t.effective ? <>{md("**② 유효 라벨 (2단계)**")}<Caption>{t.effective}</Caption></> : null}
            </div>
          </Cols>
          <Divider />
          {md("**③ 라벨 이력**")}
          {t.events ? <DataTable table={t.events} scroll /> : <Caption>사람이 남긴 라벨 이벤트가 없습니다.</Caption>}
          {md("**④ 모델 판정 이력 (4단계 추론 로그)**")}
          {t.inferences ? <DataTable table={t.inferences} scroll /> : <Caption>이 이미지에 대한 추론 기록이 없습니다.</Caption>}
          {md("**⑤ Claude 2차 판정 (3단계)**")}
          {t.reviews.length === 0 ? <Caption>2차 판정 기록이 없습니다.</Caption> : t.reviews.map((r, i) => (
            <Card key={i}>{md(r.title)}<Caption>{r.reason}</Caption></Card>
          ))}
        </>
      )}
    </>
  );
}
