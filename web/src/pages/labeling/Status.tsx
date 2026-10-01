import { useState } from "react";
import { downloadText, downloadUrl, get, post, type Count, type Table } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { useProject } from "../../components/Layout";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Bars, Button, Caption, Checkbox, Cols, DataTable, Divider, ErrorBox, Expander, Metric, Spinner, md } from "../../components/ui";
import { CodeBlock, NoImages, type Notice } from "./shared";

interface Stats {
  total: number; normal: number; defect: number; unlabeled: number; human: number; verified: number;
  unspecified_type: number; unmapped_types: number; with_roi: number; split_assigned: number;
}
interface StatusPayload { stats: Stats; defect_types?: Count[]; label_sources?: Count[]; events_total?: number; events?: Table }
interface TextFile { filename: string; mime: string; text: string }

async function saveText(name: string) {
  const file = await get<TextFile>(`/api/labeling/export/${name}`);
  downloadText(file.filename, file.text, file.mime);
}

export function StatusTab() {
  const project = useProject();
  const { data, error } = useFetch<StatusPayload>("/api/labeling/status", [project.version]);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  const s = data.stats;
  if (s.total === 0) return <NoImages />;

  return (
    <>
      <Cols n={5}>
        <Metric label="전체" value={s.total.toLocaleString()} />
        <Metric label="정상" value={s.normal.toLocaleString()} />
        <Metric label="결함" value={s.defect.toLocaleString()} />
        <Metric label="사람이 라벨" value={s.human.toLocaleString()} />
        <Metric label="확인 완료" value={s.verified.toLocaleString()} />
      </Cols>
      <Cols n={4}>
        <Metric label="유형 미지정 결함" value={s.unspecified_type.toLocaleString()} />
        <Metric label="미매핑 유형 수" value={s.unmapped_types.toLocaleString()} />
        <Metric label="ROI 지정" value={s.with_roi.toLocaleString()} />
        <Metric label="분할 배정" value={s.split_assigned.toLocaleString()} />
      </Cols>
      {s.unspecified_type ? (
        <Alert kind="info" icon="🏷️">{`결함 ${s.unspecified_type.toLocaleString()}건의 유형이 아직 미지정이다. **라벨 검수** 탭에서 지정한다.`}</Alert>
      ) : null}

      <Divider />
      <Cols n={2}>
        <div>
          {md("**결함 유형 분포 (정규화 후)**")}
          {data.defect_types && data.defect_types.length ? <Bars items={data.defect_types} /> : <Caption>결함 이미지가 없습니다.</Caption>}
        </div>
        <div>
          {md("**라벨 근거**")}
          <Bars items={data.label_sources ?? []} />
        </div>
      </Cols>

      <Divider />
      <h3>라벨 이력</h3>
      {!data.events_total ? <Caption>아직 기록된 라벨 이벤트가 없습니다.</Caption> : (
        <>
          <Caption>{`총 ${data.events_total.toLocaleString()}건 (같은 이미지의 최신 이벤트가 유효 라벨이 된다)`}</Caption>
          <DataTable table={data.events} scroll />
        </>
      )}

      <Expander title="내려받기">
        <div className="row tight">
          <Button onClick={() => void saveText("resolved")}>유효 라벨 (resolved.csv)</Button>
          {data.events_total ? <Button onClick={() => void saveText("events")}>라벨 이력 (labels.csv)</Button> : null}
        </div>
      </Expander>

      <Divider />
      <BoxExport />
    </>
  );
}

// --- 결함 박스 ----------------------------------------------------------------

interface BoxesPayload {
  summary: { boxes: number; images: number; labels: number; per_image: number };
  mask: { ready: number; without_mask: number; already: number };
  unspecified: number; adoptable: number; class_names: string[]; yolo_images: number;
  readiness: { images_with_boxes: number; boxes: number; per_class: Record<string, number>; thin_classes: string[]; splits_assigned: number; ready: boolean };
}
interface MaskResult { images: number; boxes: number; skipped_existing: number; skipped_no_mask: number; skipped_empty: number }
interface ExportResult { message: string; blocking: string; notes: string[]; root: string; command: string }

/**
 * 박스를 학습 프레임워크가 읽는 형식으로 내보낸다. 내부는 고치기 쉬운 CSV로 두고, **내보낼 때** 변환한다.
 * 변환은 언제든 다시 할 수 있지만 편집 이력은 한 번 잃으면 끝이다.
 */
function BoxExport() {
  const project = useProject();
  const { data, error, reload } = useFetch<BoxesPayload>("/api/labeling/boxes", [project.version]);
  const [notice, setNotice] = useState<Notice>(null);
  const [maskDone, setMaskDone] = useState(false);
  const [maskResult, setMaskResult] = useState<MaskResult | null>(null);
  const mask = useJob<MaskResult>((result) => { setMaskResult(result); setMaskDone(true); void reload(); });
  const [copyImages, setCopyImages] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exported, setExported] = useState<ExportResult | null>(null);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  const info = data.summary;

  const adopt = async () => {
    try {
      const out = await post<{ moved: number }>("/api/labeling/boxes/adopt");
      setNotice({ kind: "success", text: `${out.moved.toLocaleString()}건을 옮겼습니다.` });
      await reload();
    } catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
  };
  const exportDetection = async () => {
    setExporting(true);
    setExported(null);
    try { setExported(await post<ExportResult>("/api/labeling/detection/export", { copy_images: copyImages })); }
    catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
    finally { setExporting(false); }
  };

  // 결함 마스크에서 박스를 일괄로 만든다. 검출 학습에는 박스가 수백 장 필요한데 손으로만 그리면 며칠이 걸린다.
  // VisA·MVTec은 결함 픽셀 마스크를 함께 주므로 그것을 박스로 바꾸면 바로 채워진다. **덩어리마다 하나씩** 만든다.
  // 결함 픽셀 전부를 한 박스로 감싸면 실측으로 면적이 평균 2.5배가 되고, 그 박스의 절반 가까이가 배경이라
  // 모델이 배경을 결함이라고 배운다.
  // 끝난 뒤에는 접어 두되, 방금 만든 결과는 보여야 한다 — Streamlit은 rerun으로 결과가 사라졌지만 여기서는 남는다.
  const maskSection = (data.mask.ready || data.mask.already) ? (
    <Expander title={`🎭 마스크에서 박스 자동 생성 (${data.mask.ready.toLocaleString()}장 가능)`} open={!maskDone || maskResult !== null}>
      <Caption>VisA·MVTec이 함께 주는 결함 픽셀 마스크를 박스로 바꿉니다. **떨어져 있는 결함은 따로따로** 박스가 됩니다 — 전부 한 박스로 묶으면 절반이 배경이라 모델이 배경을 결함이라고 배웁니다. **이미 박스가 있는 이미지는 건드리지 않습니다.**</Caption>
      <Cols n={3}>
        <Metric label="자동 생성 가능" value={`${data.mask.ready.toLocaleString()}장`} />
        <Metric label="이미 박스 있음" value={`${data.mask.already.toLocaleString()}장`} />
        <Metric label="마스크 없음" value={`${data.mask.without_mask.toLocaleString()}장`} />
      </Cols>
      {data.unspecified ? (
        <Caption>{`⚠️ 결함 ${data.unspecified.toLocaleString()}장은 유형이 미지정입니다. 박스도 미지정으로 들어가 **"결함이 어디 있는가"만 배우는 1클래스 검출**이 됩니다. 유형별로 나누려면 **결함 유형 정규화** 탭을 먼저 거치세요.`}</Caption>
      ) : null}
      {maskResult ? (
        <>
          <Alert kind="success">{`${maskResult.images.toLocaleString()}장에 박스 ${maskResult.boxes.toLocaleString()}개를 만들었습니다 (장당 평균 ${(maskResult.boxes / Math.max(maskResult.images, 1)).toFixed(1)}개).`}</Alert>
          {maskResult.skipped_existing ? <Caption>{`이미 박스가 있어 건드리지 않은 이미지 ${maskResult.skipped_existing.toLocaleString()}장`}</Caption> : null}
          {maskResult.skipped_no_mask ? <Caption>{`마스크가 없어 건너뛴 이미지 ${maskResult.skipped_no_mask.toLocaleString()}장`}</Caption> : null}
          {maskResult.skipped_empty ? <Caption>{`마스크가 비어 있거나 잡티뿐이라 건너뛴 이미지 ${maskResult.skipped_empty.toLocaleString()}장`}</Caption> : null}
        </>
      ) : null}
      {!data.mask.ready ? <Caption>새로 만들 것이 없습니다.</Caption> : (
        <Button primary busy={mask.running} onClick={() => { setMaskResult(null); void mask.start(() => post("/api/labeling/boxes/from-masks")); }}>
          {`🎭 ${data.mask.ready.toLocaleString()}장에 박스 만들기`}
        </Button>
      )}
      <JobProgress job={mask.job} error={mask.error} />
    </Expander>
  ) : null;

  return (
    <>
      <h3>결함 박스</h3>
      <Cols n={4}>
        <Metric label="박스" value={`${info.boxes.toLocaleString()}개`} />
        <Metric label="박스가 있는 이미지" value={`${info.images.toLocaleString()}장`} />
        <Metric label="클래스" value={`${info.labels.toLocaleString()}종`} />
        <Metric label="이미지당 평균" value={`${info.per_image.toFixed(1)}개`} />
      </Cols>
      {maskSection}
      {notice ? <Alert kind={notice.kind}>{notice.text}</Alert> : null}

      {!info.boxes ? (
        <>
          <Caption>아직 박스가 없습니다. **라벨 검수** 탭에서 결함으로 판정하고 이미지 위에 그리세요.</Caption>
          {data.adoptable ? (
            <>
              <Caption>{`예전 방식으로 지정한 위치 ${data.adoptable.toLocaleString()}건이 있습니다.`}</Caption>
              <Button onClick={() => void adopt()}>📥 예전 ROI를 박스로 가져오기</Button>
            </>
          ) : null}
        </>
      ) : (
        <>
          <Caption>{"클래스 번호는 이름을 정렬한 순서로 매깁니다: " + data.class_names.map((name, i) => `${i}=${name}`).join(", ")}</Caption>
          <Cols n={3}>
            <Button block onClick={() => void saveText("boxes")}>⬇️ 박스 CSV</Button>
            <Button block title="COCO는 [x, y, 폭, 높이] 좌상단 기준. category_id는 1부터입니다." onClick={() => void saveText("coco")}>⬇️ COCO JSON</Button>
            <Button block title="이미지 한 장당 txt 한 개 + classes.txt. YOLO는 이미지 크기로 나눈 중심 좌표를 쓰므로 크기를 모르는 이미지는 빠집니다."
              onClick={() => void downloadUrl("/api/labeling/export-yolo.zip", "labels_yolo.zip")}>⬇️ YOLO (zip)</Button>
          </Cols>
          {data.yolo_images < info.images ? (
            <Caption>{`YOLO 내보내기에서 ${(info.images - data.yolo_images).toLocaleString()}장이 빠졌습니다 — manifest에 폭·높이가 없어 정규화할 수 없습니다.`}</Caption>
          ) : null}

          {/* 검출 모델 학습 폴더를 만든다. 내려받기 버튼과 달리 **디스크에 폴더를 만든다.** 학습에 쓰는 것은 라벨 몇 KB가
              아니라 이미지 수천 장이라 브라우저로 내려받을 물건이 아니다. 학습은 GPU가 있는 기계에서
              `scripts/train_detector.py`로 돌린다. */}
          {md("**지도학습 검출 모델 준비**")}
          <Caption>지금 모델(이상탐지)은 '정상과 얼마나 다른가'를 잽니다. **'스크래치 2개와 찍힘 1개'처럼 무엇이 몇 개인지**를 알려면 박스로 학습하는 검출 모델이 필요하고, 그건 GPU가 있는 기계에서 돌립니다.</Caption>
          {!data.readiness.ready ? (
            <Alert kind="info" icon="🧪">{[
              data.readiness.images_with_boxes < 50 ? `박스가 있는 이미지가 ${data.readiness.images_with_boxes.toLocaleString()}장입니다 — 검출 모델은 이 정도로 학습되지 않습니다(자릿수로 수백 장 이상).` : "",
              !data.readiness.splits_assigned ? "분할이 배정되지 않았습니다 — **데이터 분할** 탭을 먼저 실행하세요." : "",
            ].filter(Boolean).join(" ") || "아직 준비되지 않았습니다."}</Alert>
          ) : null}
          {data.readiness.thin_classes.length ? (
            <Caption>{`박스가 50개 미만인 클래스: ${data.readiness.thin_classes.join(", ")} — 이 클래스는 거의 학습되지 않습니다.`}</Caption>
          ) : null}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "0.75rem", alignItems: "center" }}>
            <Button busy={exporting} onClick={() => void exportDetection()}>📦 검출 학습 폴더 내보내기</Button>
            <Checkbox label="이미지를 복사 (링크 대신)" checked={copyImages} onChange={setCopyImages}
              help="기본은 심볼릭 링크라 용량을 두 배로 쓰지 않습니다. 폴더째 다른 기계로 옮겨 학습할 생각이면 복사해야 링크가 끊기지 않습니다." />
          </div>
          {exported ? (
            <>
              {exported.blocking
                ? <Alert kind="error" icon="🚫">{`${exported.message}\n\n${exported.blocking}`}</Alert>
                : <Alert kind="success">{exported.message}</Alert>}
              {exported.notes.map((n) => <Caption key={n}>{n}</Caption>)}
              {!exported.blocking ? <CodeBlock text={exported.command} /> : null}
            </>
          ) : null}
        </>
      )}
    </>
  );
}
