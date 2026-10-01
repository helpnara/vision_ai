import { useState } from "react";
import { post } from "../../api";
import { useJob } from "../../hooks";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Button, Caption, Checkbox, Cols, NumberInput, Select, md } from "../../components/ui";
import type { IngestOptions } from "./types";

interface SyntheticResult { out_dir: string; message: string; added: number }

export function SyntheticTab({ options, onDataChanged }: { options: IngestOptions; onDataChanged: () => void }) {
  const [categories, setCategories] = useState<string[]>(options.surface_styles.slice(0, 2));
  const [nNormal, setNNormal] = useState(40);
  const [nDefect, setNDefect] = useState(20);
  const [size, setSize] = useState("256");
  const [seed, setSeed] = useState(42);
  const [layout, setLayout] = useState<"visa" | "mvtec">("visa");
  const [overwrite, setOverwrite] = useState(true);
  const job = useJob<SyntheticResult>(() => onDataChanged());
  const result = job.job?.result ?? null;

  const toggle = (name: string, on: boolean) =>
    setCategories((cs) => (on ? [...cs, name] : cs.filter((c) => c !== name)));
  const start = () => void job.start(() => post("/api/ingest/synthetic", {
    categories, n_normal: Math.round(nNormal), n_defect: Math.round(nDefect), size: Number(size), seed: Math.round(seed), layout, overwrite,
  }));

  return (
    <>
      {md("오픈 데이터셋 다운로드는 용량이 크고 약관 동의가 필요한 경우가 많다. **수집 → 라벨링 → 학습 → 운영** 전 과정을 먼저 돌려보기 위해, MVTec AD와 동일한 폴더 구조로 가짜 표면 이미지를 생성한다.")}
      <Caption>합성 데이터는 파이프라인 배선 검증용이다. 모델 성능의 근거로 삼을 수는 없다.</Caption>

      <Cols n={3}>
        <div className="field">
          <label>표면 종류</label>
          {options.surface_styles.map((name) => (
            <Checkbox key={name} label={name} checked={categories.includes(name)} onChange={(on) => toggle(name, on)} />
          ))}
        </div>
        <NumberInput label="카테고리별 정상 이미지" value={nNormal} min={5} max={500} step={5} onChange={setNNormal} />
        <NumberInput label="카테고리별 결함 이미지" value={nDefect} min={4} max={400} step={4} onChange={setNDefect} />
      </Cols>
      <Cols n={4}>
        <Select label="이미지 크기(px)" value={size} options={options.synthetic_sizes.map(String)} onChange={setSize} />
        <NumberInput label="랜덤 시드" value={seed} min={0} max={10000} step={1} onChange={setSeed} />
        <Select<"visa" | "mvtec"> label="폴더 구조" value={layout} options={["visa", "mvtec"]} onChange={setLayout}
          help="visa: 기본 예시 데이터셋과 같은 구조 (결함 유형 폴더 없음, 마스크 제공) · mvtec: 결함 유형별 폴더 + ground_truth 마스크" />
        <Checkbox label="기존 합성 데이터 삭제 후 재생성" checked={overwrite} onChange={setOverwrite} />
      </Cols>

      <Caption>{`생성될 결함 유형: ${options.synthetic_defects.join(", ")} · 결함 픽셀 마스크를 함께 만들어 2단계에서 ROI 자동 추출을 시험할 수 있다.`}</Caption>
      {layout === "visa" ? (
        <Caption>VisA 구조는 결함 유형을 폴더로 나누지 않으므로, 등록 시 모든 결함이 `유형 미지정`으로 들어온다 — 실제 VisA와 같은 상황이며 2단계에서 유형을 지정한다.</Caption>
      ) : null}

      {categories.length === 0 ? <Alert kind="info">표면 종류를 1개 이상 선택하세요.</Alert> : (
        <>
          <Button primary busy={job.running} onClick={start}>🧪 합성 데이터 생성 후 등록</Button>
          <JobProgress job={job.job} error={job.error} />
          {result ? <Alert kind="success" icon="✅">{`${result.out_dir} 생성 완료 — ${result.message}`}</Alert> : null}
        </>
      )}
    </>
  );
}
