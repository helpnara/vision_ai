import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 개발 중에는 `npm run dev`(5173)가 화면을, `python serve.py`(8000)가 API를 맡는다.
// 빌드 결과(dist/)는 저장소에 넣어 두므로 PC에서는 Node 없이 serve.py만으로 돈다.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500, // vega 묶음이 크다. 차트가 있는 화면에서만 지연 로드한다.
  },
});
