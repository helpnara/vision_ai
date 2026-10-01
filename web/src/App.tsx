import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Ingest } from "./pages/Ingest";
import { Labeling } from "./pages/Labeling";
import { Modeling } from "./pages/Modeling";
import { Operations } from "./pages/Operations";
import { Settings } from "./pages/Settings";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="/ingest" element={<Ingest />} />
        <Route path="/labeling" element={<Labeling />} />
        <Route path="/modeling" element={<Modeling />} />
        <Route path="/operations" element={<Operations />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
