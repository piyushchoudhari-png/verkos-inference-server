# Dashboard Redesign Plan — Verkos Simulator

## 1. Problem Statement

The current dashboard (`dashboard/index.html` + `script.js`) has four critical issues:

1. **Manual file loading** — user must pick each file individually; files have no structural relationship after opening.
2. **Data redundancy & inconsistency** — `outputs.json` and `samples.csv` carry identical `MetricSample` data; `gpu.csv` uses CSV while `outputs.json` is JSON; the frontend must parse two different formats for the same run.
3. **No frame images or bounding-box visualization** — frames are discarded after inference; the model's spatial detections are unverifiable.
4. **Limited scope for a toy HTML page** — the feature set required (bbox canvas overlays, multi-chart dashboard, PDF export) warrants a proper build tool.

---

## 2. Target Architecture

```
sim run
└── results/
    └── {run_id}/             ← one folder per run
        ├── outputs.json      ← list[MetricSample]  (canonical, unchanged)
        ├── gpu.json          ← list[GpuSample]      (replaces gpu.csv)
        ├── engine.json       ← list[EngineSample]   (replaces engine.csv)
        ├── summary.txt       ← human-readable recap (unchanged)
        └── frames/
            └── feed{NN}_frame{NNNN}.jpg   ← saved at inference time

dashboard/                    ← React + Vite SPA
├── index.html
├── package.json
├── vite.config.ts
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── types.ts
    ├── lib/
    │   ├── parseDetections.ts
    │   ├── parseJson.ts
    │   └── aggregate.ts
    └── components/
        ├── RunLoader.tsx
        ├── RunOverview.tsx
        ├── FrameExplorer.tsx
        ├── FrameCard.tsx
        ├── BBoxCanvas.tsx
        ├── LatencyChart.tsx
        ├── GpuChart.tsx
        ├── EngineChart.tsx
        ├── ErrorPanel.tsx
        └── ExportButton.tsx
```

---

## 3. Backend Changes

### 3.1 Frame Saving

Add frame persistence in `src/feed_runner.py` inside `run_feed()`, immediately after extracting each frame and **before** dispatching inference.

**Location:** `run_feed()` — receives `config` (has `output_dir`) and `run_id`.

**Frames directory:** `{config.output_dir}/{run_id}/frames/`  
**Filename convention:** `feed{feed_id:02d}_frame{frame_index:04d}.jpg`  
— `feed_id` disambiguates frames across concurrent feeds (each feed has its own 0-based `frame_index`).  
— JPEG at quality 90; size is small relative to model inference latency.

The frames directory must be created before the first write. A single `Path.mkdir(parents=True, exist_ok=True)` at the start of `run_feed()` is sufficient (the run dir was already created by `reporter.write_results()`, but frames is a new subdir, so create it eagerly).

`config` already has `output_dir` and `run_id` is passed as a parameter. No new config fields needed.

### 3.2 Output Format: CSV → JSON

**Drop** `samples.csv` — fully redundant with `outputs.json`.  
**Replace** `gpu.csv` → `gpu.json` (list of `GpuSample` dicts via `model_dump()`).  
**Replace** `engine.csv` → `engine.json` (list of `EngineSample` dicts).

**Changes confined to `src/reporter.py`:**
- Remove `_write_csv` calls for samples and gpu/engine.
- Add `_write_json(records, path)` helper: `path.write_text(json.dumps([r.model_dump() for r in records], indent=2))`.
- Update `paths` dict: `paths["gpu"]` → `gpu.json`, `paths["engine"]` → `engine.json`.
- Remove `samples` key from paths entirely (data is already in `outputs.json`).

**No changes to `runner.py` or `__main__.py`** — they only log `arts.paths` keys and don't depend on filenames.

---

## 4. Frontend: React + Vite

### 4.1 Tech Stack

| Concern | Choice | Reason |
|---|---|---|
| Build | Vite | Fast HMR, native ESM, minimal config |
| UI | React 18 + TypeScript | Component model needed for canvas overlays + conditional rendering |
| Styling | Tailwind CSS v4 | Utility-first, no custom CSS needed at this scale |
| Charts | Recharts | React-native, declarative, good TypeScript types |
| PDF export | `html2pdf.js` | Captures canvas elements correctly; `window.print()` does not |
| File access | File System Access API (`showDirectoryPicker`) | Reads the entire run folder in one action; no per-file pickers |

`showDirectoryPicker()` is supported in Chrome, Edge, and Opera. For Firefox: graceful fallback to three separate file inputs (`outputs.json`, `gpu.json`, `frames/` folder via `<input type="file" webkitdirectory>`).

### 4.2 Run Loading Flow

```
User clicks "Open Run Folder"
  → showDirectoryPicker()                      ← user selects results/{run_id}/
  → read outputs.json  (required)
  → read gpu.json      (optional, may not exist in openrouter mode)
  → read engine.json   (optional, local mode only)
  → read summary.txt   (optional, for overview header)
  → enumerate frames/  (optional; build map: "feed00_frame0005" → FileHandle)
  → setState({ outputs, gpu, engine, summary, frameHandles })
```

All reads are async; show a loading indicator. If `outputs.json` is missing or unparseable, show an error and stop.

### 4.3 Detection Parsing (`lib/parseDetections.ts`)

The model returns `output_text` as free-form text. Observed in real run data:

1. **Output is often wrapped in a markdown code fence:**  
   ` ```json\n{...}\n``` ` — strip with regex before `JSON.parse`.

2. **Three top-level structures:**
   - `{ "detections": [...] }` — each item has `event`/`bbox`/`confidence`
   - `{ "events": [...] }` — each item has `type`/`count`/`bbox`
   - `{ "person": [...], "vehicle": [...], ... }` — keys are event names, values are bbox arrays

3. **Bbox field name variants:** `"bbox"` or `"bounding_boxes"`

4. **Bbox value variants:**
   - Single box: `[x1, y1, x2, y2]` (flat array of 4 ints)
   - Multiple boxes: `[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]`
   - Both `"bbox"` (pixel) and `"normalized_bbox"` (0–1) present simultaneously

5. **Coordinate space:** Despite the prompt requesting normalized [0,1], the model returns **pixel integers** relative to the original resolution stored in `resolution_w`/`resolution_h` of `MetricSample`.

**Canonical output type:**
```typescript
interface Detection {
  event: string;
  count: number;
  boxes: [number, number, number, number][];  // [[x1,y1,x2,y2]] pixel coords
  confidence?: number;
}

interface ParsedOutput {
  detections: Detection[];
  notes?: string;
  rawError?: string;  // set if JSON.parse failed
}
```

**Parser algorithm:**
1. Strip leading/trailing whitespace and markdown fences (`/^```json\s*/`, `/\s*```$/`).
2. `JSON.parse(cleaned)` — on failure set `rawError` and return.
3. Detect top-level structure type.
4. Normalize all items into `Detection[]`:
   - Extract event name from `event` or `type` key (or the dict key itself).
   - Extract boxes from `bbox` or `bounding_boxes` field.
   - Normalize single `[x1,y1,x2,y2]` to `[[x1,y1,x2,y2]]`.
5. Validate each box: must be a 4-element numeric array; skip malformed entries and flag them in console.
6. **Do not** use `normalized_bbox` if pixel `bbox` is present — pixel coords are more reliable based on observed data.

**Bounding box scaling for display:**
```
scaleX = canvasDisplayWidth  / resolution_w
scaleY = canvasDisplayHeight / resolution_h
displayBox = [x1*scaleX, y1*scaleY, x2*scaleX, y2*scaleY]
```
Canvas display size is set in CSS; actual pixel dimensions of the `<canvas>` element must match the CSS size to avoid DPI blur (`canvas.width = canvas.offsetWidth * devicePixelRatio`).

### 4.4 Component Breakdown

#### `RunLoader`
- "Open Run Folder" button → `showDirectoryPicker()`.
- Firefox fallback: three drag-and-drop zones.
- Emits `RunData` to parent on successful load.

#### `RunOverview`
- Stat cards: total frames, ok/error counts, error rate %, latency p50/p95/p99/mean, TTFT p50, throughput fps, run duration, model name, inference mode.
- Computed client-side from `outputs` array (mirrors `aggregate()` in Python — keeps numbers consistent).
- Source of truth: `outputs.json` only, not `summary.txt` (avoids double-source drift).

#### `FrameExplorer`
- Responsive grid of `FrameCard` components.
- Filter bar: feed selector, status (ok/error), event type checkboxes.
- Sort: by frame index (default), latency (desc), time.
- Virtual scrolling if frame count > 100 (use `react-window`).

#### `FrameCard`
- Top: `BBoxCanvas` (image + overlaid boxes).
- Bottom left: frame index, feed id, status badge, latency, TTFT, token counts.
- Bottom right: detection list (event name + count + confidence if available).
- Click to expand: full `output_text` (raw and parsed), error message if applicable.
- Error frames: red border, no canvas (image may be missing), show `error_msg`.

#### `BBoxCanvas`
- `<canvas>` element overlaid on `<img>` using CSS `position: absolute`.
- Loads frame image from `FileSystemFileHandle` via `createObjectURL(await handle.getFile())`.
- On image load: draw boxes with `strokeRect`, label each detection.
- Color-codes by event type (person=blue, vehicle=orange, fire=red, intrusion=purple, other=gray).
- Legend overlay in top-left corner.
- If frame file not found: shows placeholder with "frame not saved" message.

#### `LatencyChart`
- Recharts `ComposedChart`: scatter plot of latency_s per frame (x = t0_epoch relative to run start, y = latency_s).
- Reference lines for p50/p95.
- Color by feed_id.
- Hover tooltip: frame index, feed, latency, status.

#### `GpuChart`
- Recharts `LineChart` with dual Y-axes: VRAM GB (left), GPU util % (right).
- One line per GPU index.
- X-axis: seconds from run start.
- Only rendered when `gpu.json` is present.

#### `EngineChart`
- Recharts `AreaChart`: stacked areas for `num_running`, `num_waiting`, `num_swapped`.
- Second chart: `gpu_kv_cache_usage_pct` over time.
- Only rendered when `engine.json` is present (local mode only).

#### `ErrorPanel`
- Table of error frames: frame index, feed, error_msg, error class.
- Bar chart: error count by class (oom / timeout / cuda / model / other).
- Hidden when error count = 0.

#### `ExportButton`
- Uses `html2pdf.js`.
- Captures the entire `#dashboard-root` div as a multi-page PDF.
- Canvas elements are captured correctly by html2pdf (converts via `toDataURL`).
- Button in sticky header, always visible.
- Options: A4 landscape, 2× scale for sharpness.

### 4.5 TypeScript Types (`types.ts`)

```typescript
// mirrors Python MetricSample
interface MetricSample {
  run_id: string;
  feed_id: number;
  frame_index: number;
  t0_epoch: number;
  latency_s: number;
  ttft_s: number | null;
  prompt_tokens: number;
  completion_tokens: number;
  resolution_w: number;
  resolution_h: number;
  model: string;
  inference_mode: string;
  status: "ok" | "error";
  error_msg: string | null;
  output_text: string | null;
}

// mirrors Python GpuSample
interface GpuSample {
  run_id: string;
  t_epoch: number;
  gpu_index: number;
  memory_used_bytes: number;
  memory_total_bytes: number;
  torch_allocated_bytes: number;
  gpu_util_pct: number;
  power_w: number | null;
  temperature_c: number | null;
}

// mirrors Python EngineSample
interface EngineSample {
  run_id: string;
  t_epoch: number;
  num_running: number;
  num_waiting: number;
  num_swapped: number;
  gpu_kv_cache_usage_pct: number;
  cpu_kv_cache_usage_pct: number;
  num_preemption_total: number;
}

interface RunData {
  runId: string;
  outputs: MetricSample[];
  gpu: GpuSample[];          // empty if not present
  engine: EngineSample[];    // empty if not present
  summary: string | null;
  frameHandles: Map<string, FileSystemFileHandle>;  // key: "feed00_frame0005"
}
```

---

## 5. Data Accuracy Guarantees

### Bounding Box Accuracy
- **Use `resolution_w`/`resolution_h` from `MetricSample`** (not hardcoded) for scaling — the simulator stores the actual image dimensions per frame.
- Always validate: `0 ≤ x1 < x2 ≤ resolution_w` and `0 ≤ y1 < y2 ≤ resolution_h`. Boxes failing this check are drawn but flagged with a warning overlay.
- `normalized_bbox` is only used if `bbox` is absent, since pixel coords are observed to be more consistent.

### Frame–Output Alignment
- `outputs.json` entry identified by `(feed_id, frame_index)`.
- Frame filename constructed deterministically: `feed{feed_id:02d}_frame{frame_index:04d}.jpg`.
- If the frame file is missing (run without frame saving, or partial run), the detection data still renders — only the image is absent.

### Metric Accuracy
- All aggregations (p50/p95/p99, throughput fps, error rate) are recomputed in-browser from `outputs.json` using the same algorithm as Python's `aggregate()` in `src/metrics.py`. Do **not** parse `summary.txt` for numbers — it is display-only.
- Latency timeline uses `t0_epoch` for x-axis alignment across feeds, giving true wall-clock ordering.

---

## 6. PDF Export

`html2pdf.js` captures the live DOM including canvas elements (via `toDataURL`). Configuration:

```typescript
html2pdf()
  .set({
    margin: 10,
    filename: `verkos_run_${runId}.pdf`,
    image: { type: "jpeg", quality: 0.92 },
    html2canvas: { scale: 2, useCORS: true },
    jsPDF: { unit: "mm", format: "a4", orientation: "landscape" },
    pagebreak: { mode: ["avoid-all", "css"] },
  })
  .from(document.getElementById("dashboard-root"))
  .save();
```

Add `pagebreak: avoid` CSS class to `FrameCard` to prevent mid-card page splits.

---

## 7. Implementation Steps

### Step 1 — Backend: Frame Saving
- `src/feed_runner.py`: create frames dir at start of `run_feed()`; save each `(feed_id, frame_index, image)` as JPEG before dispatching inference task.
- Import: `from pathlib import Path` (already available); `image.save(path, format="JPEG", quality=90)` via PIL (already a dependency).

### Step 2 — Backend: Drop CSVs, Add JSON outputs
- `src/reporter.py`: remove `_write_csv` usage for gpu/engine/samples; add `_write_json`; update path keys.
- No other files change.

### Step 3 — Scaffold React + Vite App
- Replace `dashboard/` contents with Vite project: `npm create vite@latest dashboard -- --template react-ts`.
- Add dependencies: `recharts`, `html2pdf.js`, `tailwindcss`.
- Keep old `index.html`/`script.js` in git history — do not archive them separately.

### Step 4 — Types + Data Layer
- `src/types.ts`: define all interfaces (Step 4.5 above).
- `src/lib/parseDetections.ts`: implement the parser (all 5 bbox variant handlers).
- `src/lib/aggregate.ts`: reimplement Python's `aggregate()` in TypeScript.

### Step 5 — RunLoader Component
- Implement `showDirectoryPicker()` flow.
- Implement Firefox fallback.
- Parse and validate all loaded files; emit `RunData`.

### Step 6 — Frame Explorer + BBoxCanvas
- `FrameCard`: layout, status badges, detection list.
- `BBoxCanvas`: canvas overlay, scaling, color legend.
- `FrameExplorer`: grid, filters, sorting.

### Step 7 — Charts
- `LatencyChart`, `GpuChart`, `EngineChart` with Recharts.

### Step 8 — Overview + Error Panel + Export
- `RunOverview` stat cards.
- `ErrorPanel` table + chart.
- `ExportButton` with html2pdf.

### Step 9 — Polish
- Responsive layout (grid collapses on narrow viewport).
- Loading skeleton screens.
- Empty-state handling (no GPU data, no frames, error-only run).
- `README.md` update: how to run the dashboard (`cd dashboard && npm run dev`).

---

## 8. File Changes Summary

| File | Change |
|---|---|
| `src/feed_runner.py` | Add frame saving in `run_feed()` |
| `src/reporter.py` | Drop `samples.csv`; replace `gpu.csv`/`engine.csv` with `.json`; drop `_write_csv` |
| `dashboard/index.html` | Replaced (Vite entry point) |
| `dashboard/script.js` | Deleted |
| `dashboard/package.json` | New |
| `dashboard/vite.config.ts` | New |
| `dashboard/src/**` | New React app |

No changes to `src/metrics.py`, `src/runner.py`, `src/config.py`, `src/__main__.py`, or `src/engine*.py`.

---

