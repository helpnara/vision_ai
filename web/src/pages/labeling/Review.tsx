import { useEffect, useState } from "react";
import { imageUrl, post } from "../../api";
import { useFetch } from "../../hooks";
import { useProject } from "../../components/Layout";
import { RoiPicker, zoomNote, zoomTo, type Box } from "../../components/RoiPicker";
import { Alert, Button, Caption, ErrorBox, Radio, Select, Spinner, TextInput, md } from "../../components/ui";
import { DEFECT_TYPE_UNSPECIFIED, LABEL_DEFECT, LABEL_NORMAL, LABEL_UNLABELED, NoImages, defectChoiceLabel, defectTypeLabel, labelKo, type Notice, type Overview } from "./shared";

const ALL_CATEGORIES = "(전체)";

const MODE_MASK = "마스크에서 자동 추출";
const MODE_DRAG = "이미지에서 직접 그리기";
const MODE_NONE = "지정 안 함";
type RoiMode = typeof MODE_MASK | typeof MODE_DRAG | typeof MODE_NONE;

const ROI_DISPLAY_WIDTH = 640;
/** 영역 지정 화면에서 이미지를 그릴 폭(px). 좌표는 원본 픽셀 기준으로 돌려준다 (예전 ui.ROI_DISPLAY_WIDTH). */

const SOURCE_HUMAN = "사람";
const SOURCE_MASK = "마스크";

interface Queue { total_images: number; categories: string[]; ids: string[] }
interface Item {
  image_id: string; category: string; path: string; readable: boolean; width: number | null; height: number | null;
  label: string; label_ko: string; defect_type: string; defect_type_label: string;
  source: string; label_source: string; raw_defect_type: string; verified: boolean;
  auto_roi: [number, number, number, number] | null;
  boxes: Drawn[];
}
interface Drawn extends Box { label: string; source: string }

const LABEL_OPTIONS: readonly string[] = [LABEL_NORMAL, LABEL_DEFECT, LABEL_UNLABELED];

/** 예전 `boxes.retype_unspecified` — 아직 유형이 안 붙은 박스에 지금 고른 유형을 붙인다.
 *  박스를 먼저 그리고 유형을 나중에 고르는 순서가 자연스러운데, 그대로 두면 유형을 골라도
 *  박스는 "유형 미지정"으로 저장된다. 이미 유형이 붙은 박스는 건드리지 않는다. */
function retyped(drawn: Drawn[], defectType: string | null): Drawn[] {
  if (!defectType) return drawn;
  return drawn.map((b) => (b.label === DEFECT_TYPE_UNSPECIFIED ? { ...b, label: defectType } : b));
}

const boxesQuery = (drawn: Drawn[]) => drawn.map((b) => `${b.x},${b.y},${b.w},${b.h},${b.label}`).join(";");

export function ReviewTab({ overview }: { overview: Overview }) {
  const project = useProject();
  const [mode, setMode] = useState<string>("unspecified");
  const [category, setCategory] = useState<string>(ALL_CATEGORIES);
  // 커서는 모드·카테고리별로 따로 기억한다 (예전 st.session_state["p2_cursor::mode::category"]).
  const [cursors, setCursors] = useState<Record<string, number>>({});
  const [saves, setSaves] = useState(0);
  const [notice, setNotice] = useState<Notice>(null);

  const queueUrl = `/api/labeling/queue?mode=${encodeURIComponent(mode)}&category=${encodeURIComponent(category)}`;
  const { data: queue, error, reload } = useFetch<Queue>(queueUrl, [project.version]);

  if (error) return <ErrorBox error={error} />;
  if (!queue) return <Spinner />;
  if (queue.total_images === 0) return <NoImages text="수집된 이미지가 없습니다. 1단계에서 먼저 등록하세요." />;

  const cursorKey = `${mode}::${category}`;
  const total = queue.ids.length;
  const cursor = total ? Math.min(cursors[cursorKey] ?? 0, total - 1) : 0;
  const setCursor = (n: number) => setCursors({ ...cursors, [cursorKey]: n });
  const modeLabel = (k: string) => overview.queue_modes.find((m) => m.key === k)?.label ?? k;

  const afterSave = () => {
    setSaves((n) => n + 1);
    // 라벨을 기록하면 큐에서 빠지는 모드는 커서를 그대로 둔다 — 다음 항목이 그 자리로 올라온다.
    if (!overview.draining_modes.includes(mode)) setCursor(Math.min(cursor + 1, total - 1));
    void reload();
  };

  return (
    <>
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "0.75rem" }}>
        <Select label="검수 대상" value={mode} options={overview.queue_modes.map((m) => m.key)} labels={modeLabel} onChange={setMode} />
        <Select label="카테고리" value={category} options={queue.categories} onChange={setCategory} />
      </div>
      {notice ? <Alert kind={notice.kind} icon={notice.icon}>{notice.text}</Alert> : null}
      {total === 0 ? (
        <Alert kind="success" icon="🎉">이 조건에 검수할 이미지가 없습니다.</Alert>
      ) : (
        <>
          <div className="progress"><div className="bar"><div style={{ width: `${((cursor + 1) / total) * 100}%` }} /></div>
            <div className="text">{cursor + 1} / {total}</div></div>
          <ReviewItem
            key={`${queue.ids[cursor]}::${saves}`}
            imageId={queue.ids[cursor]} overview={overview}
            onSaved={afterSave}
            onSkip={() => setCursor((cursor + 1) % total)}
            onError={(text) => setNotice({ kind: "error", text })}
          />
          <div className="row" style={{ marginTop: "0.75rem" }}>
            <Button block onClick={() => setCursor((cursor - 1 + total) % total)}>◀ 이전</Button>
            <Button block onClick={() => setCursor((cursor + 1) % total)}>다음 ▶</Button>
          </div>
        </>
      )}
    </>
  );
}

function ReviewItem({ imageId, overview, onSaved, onSkip, onError }: {
  imageId: string; overview: Overview; onSaved: () => void; onSkip: () => void; onError: (text: string) => void;
}) {
  const { data: item, error } = useFetch<Item>(`/api/labeling/item/${encodeURIComponent(imageId)}`);
  const { data: size } = useFetch<{ width: number; height: number }>(`/api/files/image/${encodeURIComponent(imageId)}/size`);
  const [label, setLabel] = useState<string | null>(null);
  const [type, setType] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [roiMode, setRoiMode] = useState<RoiMode | null>(null);
  const [boxes, setBoxes] = useState<Drawn[] | null>(null);
  const [pending, setPending] = useState<Box | null>(null);
  const [view, setView] = useState<Box | null>(null);
  const [busy, setBusy] = useState(false);

  // 처음 열 때는 **이미 저장된 박스**를 불러온다. 저장된 것을 안 보여주면 다시 열었을 때
  // 빈 화면이 나와 "지워졌나?" 하게 된다.
  useEffect(() => {
    if (!item) return;
    setLabel(LABEL_OPTIONS.includes(item.label) ? item.label : LABEL_UNLABELED);
    // 유형이 아직 정해지지 않은 이미지는 기본 선택을 두지 않는다.
    // 기본값을 두면 그냥 저장을 눌렀을 때 엉뚱한 유형이 붙는다.
    setType(overview.defect_types.some((t) => t.key === item.defect_type) ? item.defect_type : null);
    setBoxes(item.boxes);
  }, [item, overview]);

  if (error) return <ErrorBox error={error} />;
  if (!item || label === null || boxes === null) return <Spinner />;

  const isDefect = label === LABEL_DEFECT;
  const width = size?.width ?? item.width ?? 1;
  const height = size?.height ?? item.height ?? 1;
  const options: RoiMode[] = item.auto_roi ? [MODE_MASK, MODE_DRAG, MODE_NONE] : [MODE_DRAG, MODE_NONE];
  const choice: RoiMode = roiMode && options.includes(roiMode) ? roiMode : options[0];

  // 그릴 박스 목록 (예전 _box_editor의 반환값). 화면은 그리지 않고 목록만 관리한다.
  let drawn: Drawn[] = [];
  if (isDefect && item.readable) {
    if (choice === MODE_MASK && item.auto_roi) {
      const [x, y, w, h] = item.auto_roi;
      drawn = [{ x, y, w, h, label: type ?? DEFECT_TYPE_UNSPECIFIED, source: SOURCE_MASK }];
    } else if (choice === MODE_DRAG) {
      drawn = retyped(boxes, type);
    }
  }
  const needsType = isDefect && !type;
  const fullView: Box = { x: 0, y: 0, w: width, h: height };
  const currentView = view ?? fullView;
  const zoomed = currentView.w !== width || currentView.h !== height;

  const save = async () => {
    setBusy(true);
    try {
      await post("/api/labeling/save", {
        image_id: item.image_id, label, defect_type: isDefect ? type : null,
        boxes: isDefect ? drawn : [], note,
      });
      onSaved();
    } catch (e) { onError((e as Error).message); setBusy(false); }
  };

  const addBox = () => {
    if (!pending) return;
    setBoxes([...boxes, { ...pending, label: type ?? DEFECT_TYPE_UNSPECIFIED, source: SOURCE_HUMAN }]);
    setPending(null);
  };

  const preview = isDefect ? drawn : [];
  const sizeText = `${item.width ?? "?"}×${item.height ?? "?"}`;
  const note_ = zoomNote(currentView, ROI_DISPLAY_WIDTH);

  return (
    <div className="review-grid" style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: "1rem", alignItems: "start" }}>
      <div style={{ minWidth: 0 }}>
        {!item.readable ? (
          <Alert kind="error">{`이미지를 읽을 수 없습니다: ${item.path}`}</Alert>
        ) : choice === MODE_DRAG && isDefect ? (
          <>
            <div style={{ overflowX: "auto" }}>
              <RoiPicker
                src={imageUrl(item.image_id, 0)} width={width} height={height} view={currentView}
                pending={pending} saved={drawn.map((b) => ({ ...b, label: defectTypeLabel(overview, b.label) }))}
                onChange={setPending} displayWidth={ROI_DISPLAY_WIDTH}
              />
            </div>
            <Caption>{`${item.category} · ${item.image_id} · ${width}×${height} — **드래그해 영역을 지정**하고, 지정한 영역 **안쪽을 끌면 위치를 옮길 수 있습니다.** 여러 개면 하나씩 그려 **＋ 박스 추가**를 누르세요.`}</Caption>
            {/* 작은 결함을 위한 확대 (G16). VisA PCB 결함은 이미지의 0.57%다. 1404px 이미지를 640px로 줄여
                그리면 결함이 화면에서 몇 픽셀에 불과해 보이지도, 그려지지도 않는다. 조작은 늘리지 않는다 —
                이미 배운 드래그를 그대로 쓴다. 대충 끌고 확대를 누르면 그 둘레로 확대되고, 그 안에서 다시 그린다. */}
            <div className="row">
              <Button block disabled={!pending} title="대충 끌어 놓고 이 버튼을 누르면 그 둘레로 확대됩니다. 그 안에서 다시 그리세요."
                onClick={() => { if (pending) { setView(zoomTo(pending, width, height)); setPending(null); } }}>🔍 지정한 영역으로 확대</Button>
              <Button block disabled={!zoomed} onClick={() => { setView(null); setPending(null); }}>🖼️ 전체 보기</Button>
              <div style={{ flex: "1.5 1 0" }}>
                <Caption>{zoomed
                  ? `확대 중 — ${note_} · 보는 범위 ${currentView.w}×${currentView.h}`
                  : currentView.w > ROI_DISPLAY_WIDTH ? `${note_} — 이보다 작은 결함은 확대해야 그릴 수 있습니다.` : note_}</Caption>
              </div>
            </div>
          </>
        ) : (
          <>
            <img
              src={`/api/labeling/preview/${encodeURIComponent(item.image_id)}?boxes=${encodeURIComponent(boxesQuery(preview))}`}
              alt={item.image_id} style={{ width: "100%", height: "auto", display: "block", borderRadius: 6, border: "1px solid var(--border)" }}
            />
            <Caption>{`${item.category} · ${item.image_id} · ${sizeText}`}</Caption>
            {preview.length ? <Caption>{`박스 ${preview.length}개`}</Caption> : null}
          </>
        )}
      </div>

      <div style={{ minWidth: 0 }}>
        {md(`**현재 라벨**: ${item.label_ko} / ${item.defect_type_label}`)}
        <Caption>{`출처 \`${item.source}\` · 라벨 근거 \`${item.label_source}\` · 원본 유형 \`${item.raw_defect_type}\` · 확인 ${item.verified ? "✅" : "미확인"}`}</Caption>

        <Radio label="판정" value={label} options={LABEL_OPTIONS} labels={(v) => labelKo(overview, v)} onChange={setLabel} />
        <Select label="결함 유형" value={type} options={overview.defect_types.map((t) => t.key)} labels={defectChoiceLabel(overview)}
          placeholder="결함 유형을 선택하세요" disabled={!isDefect} onChange={setType} />

        {isDefect && item.readable ? (
          <>
            {md("**결함 위치 (박스)**")}
            <Radio label="" value={choice} options={options} onChange={(v) => { setRoiMode(v); setPending(null); }} />
            {choice === MODE_MASK && item.auto_roi ? (
              <Caption>{`ground truth 마스크에서 자동 추출: (${item.auto_roi.join(", ")})`}</Caption>
            ) : null}
            {choice === MODE_DRAG ? (
              <>
                <div className="row">
                  <Button block disabled={!pending} onClick={addBox} style={{ flex: "2 1 0" }}>＋ 박스 추가</Button>
                  <Button block disabled={!boxes.length} onClick={() => setBoxes([])}>모두 지우기</Button>
                </div>
                {pending
                  ? <Caption>{`그린 영역 — x ${pending.x} · y ${pending.y} · 폭 ${pending.w} · 높이 ${pending.h}`}</Caption>
                  : <Caption>왼쪽 이미지 위에서 드래그한 뒤 **＋ 박스 추가**를 누르세요.</Caption>}
                {drawn.map((b, i) => (
                  <div key={i} className="row" style={{ alignItems: "center" }}>
                    <Caption>{`**${i + 1}.** ${defectTypeLabel(overview, b.label)} — ${b.w}×${b.h} @ (${b.x}, ${b.y})`}</Caption>
                    <Button small style={{ flex: "0 0 auto" }} onClick={() => setBoxes(boxes.filter((_, j) => j !== i))}>✕</Button>
                  </div>
                ))}
                {!drawn.length ? <Caption>아직 박스가 없습니다. 결함으로 저장하면 위치 없이 기록됩니다.</Caption> : null}
              </>
            ) : null}
          </>
        ) : null}

        <TextInput label="메모 (선택)" value={note} onChange={setNote} />
        {needsType ? <Caption>⚠️ 결함으로 판정했으면 유형을 선택해야 저장할 수 있다.</Caption> : null}
        <div className="row">
          <Button primary block busy={busy} disabled={needsType} onClick={() => void save()}>💾 저장하고 다음</Button>
          <Button block onClick={onSkip}>⏭️ 건너뛰기</Button>
        </div>
      </div>
    </div>
  );
}
