/** `/api/ingest/options` — 화면이 처음 그릴 때 필요한 선택지 전부. */
export interface IngestOptions {
  has_data: boolean;
  default_dataset: { key: string; name: string; license: string; layout: string };
  layouts: string[];
  layout_help: Record<string, string>;
  labels: { key: string; label: string }[];
  defect_types: { key: string; label: string }[];
  image_extensions: string[];
  video_extensions: string[];
  surface_styles: string[];
  synthetic_defects: string[];
  synthetic_sizes: number[];
  model_inputs: { name: string; px: number }[];
  safe_px: number;
  floor_px: number;
  default_min_change: number;
  default_frames_per_object: number;
}

export interface IngestResultBody { message: string; added: number; duplicates: number; failed: string[]; n_failed: number }
