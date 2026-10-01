import { useEffect, useMemo, useState } from "react";
import { post } from "../../api";
import { Gallery } from "../../components/Gallery";
import { Alert, Button, Caption, Cols, Select, TextInput, md } from "../../components/ui";
import type { IngestOptions, IngestResultBody } from "./types";

export function UploadTab({ options, onDataChanged }: { options: IngestOptions; onDataChanged: () => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [category, setCategory] = useState("");
  const [label, setLabel] = useState(options.labels[0].key);
  const [defectType, setDefectType] = useState(options.defect_types[0].key);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<IngestResultBody | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 미리보기는 브라우저 안에서 만든다 — 서버에 올리기 전이라 왕복할 것이 없다.
  const previews = useMemo(() => files.slice(0, 5).map((f) => ({ src: URL.createObjectURL(f), caption: f.name })), [files]);
  useEffect(() => () => previews.forEach((p) => URL.revokeObjectURL(p.src)), [previews]);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      form.append("category", category);
      form.append("label", label);
      form.append("defect_type", label === "defect" ? defectType : "none");
      setResult(await post<IngestResultBody>("/api/ingest/upload", form));
      onDataChanged();
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  return (
    <>
      {md("일상에서 직접 촬영한 이미지를 올린다. 오픈 데이터셋으로 학습한 모델이 실제 생활 이미지에서 어떻게 동작하는지 확인하는 검증셋으로 쓰기 좋다.")}
      <div className="file-drop">
        <label>이미지 선택 (여러 장 가능)
          <input type="file" multiple accept={options.image_extensions.join(",")}
            onChange={(e) => { setFiles(Array.from(e.target.files ?? [])); setResult(null); }} />
        </label>
      </div>
      <Cols n={3}>
        <TextInput label="카테고리" value={category} onChange={setCategory} placeholder="mug, phone_case, wood_table ..."
          help="같은 종류의 물건끼리 묶는 이름. 비우면 uncategorized로 등록된다." />
        <Select label="라벨" value={label} options={options.labels.map((l) => l.key)} labels={(k) => options.labels.find((l) => l.key === k)?.label ?? k}
          onChange={setLabel} help="지금 모르면 미라벨로 두고 2단계 라벨링에서 지정한다." />
        <Select label="결함 유형" value={defectType} options={options.defect_types.map((d) => d.key)}
          labels={(k) => options.defect_types.find((d) => d.key === k)?.label ?? k} onChange={setDefectType} disabled={label !== "defect"} />
      </Cols>

      {files.length === 0 ? null : (
        <>
          <Caption>{`선택된 파일 ${files.length}건`}</Caption>
          <Gallery items={previews} cols={5} />
          <Button primary busy={busy} onClick={() => void submit()}>📥 업로드 이미지 등록</Button>
          {error ? <Alert kind="error">{error}</Alert> : null}
          {result ? (
            <>
              <Alert kind="success" icon="✅">{result.message}</Alert>
              {result.n_failed ? <Alert kind="warning">{`이미지로 읽을 수 없는 파일: ${result.failed.join(", ")}`}</Alert> : null}
            </>
          ) : null}
        </>
      )}
    </>
  );
}
