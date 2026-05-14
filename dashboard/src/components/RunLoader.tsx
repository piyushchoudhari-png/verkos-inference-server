import { useState, useRef } from 'react';
import type { RunData, MetricSample, GpuSample, EngineSample } from '../types';

interface Props {
  onLoad: (data: RunData) => void;
}

async function readJsonFile(handle: FileSystemFileHandle): Promise<unknown> {
  const file = await handle.getFile();
  return JSON.parse(await file.text());
}

async function readTextFile(handle: FileSystemFileHandle): Promise<string> {
  const file = await handle.getFile();
  return file.text();
}

async function loadViaDirectoryPicker(onLoad: Props['onLoad'], setError: (e: string) => void, setLoading: (v: boolean) => void) {
  setLoading(true);
  setError('');
  try {
    const dirHandle = await (window as Window & typeof globalThis & { showDirectoryPicker: () => Promise<FileSystemDirectoryHandle> }).showDirectoryPicker();
    const runId = dirHandle.name;

    let outputs: MetricSample[] = [];
    let gpu: GpuSample[] = [];
    let engine: EngineSample[] = [];
    let summary: string | null = null;
    const frameHandles = new Map<string, FileSystemFileHandle>();

    // required
    try {
      const h = await dirHandle.getFileHandle('outputs.json');
      outputs = (await readJsonFile(h)) as MetricSample[];
    } catch {
      setError('outputs.json not found or invalid in selected folder');
      setLoading(false);
      return;
    }

    // optional
    try {
      const h = await dirHandle.getFileHandle('gpu.json');
      gpu = (await readJsonFile(h)) as GpuSample[];
    } catch { /* absent in openrouter mode */ }

    try {
      const h = await dirHandle.getFileHandle('engine.json');
      engine = (await readJsonFile(h)) as EngineSample[];
    } catch { /* absent in openrouter mode */ }

    try {
      const h = await dirHandle.getFileHandle('summary.txt');
      summary = await readTextFile(h);
    } catch { /* optional */ }

    // enumerate frames/
    try {
      const framesDir = await dirHandle.getDirectoryHandle('frames');
      for await (const [name, handle] of framesDir.entries()) {
        if (handle.kind === 'file' && name.endsWith('.jpg')) {
          const key = name.replace('.jpg', '');
          frameHandles.set(key, handle as FileSystemFileHandle);
        }
      }
    } catch { /* no frames directory */ }

    onLoad({ runId, outputs, gpu, engine, summary, frameHandles });
  } catch (e) {
    if ((e as DOMException).name !== 'AbortError') {
      setError(String(e));
    }
  } finally {
    setLoading(false);
  }
}

// Firefox fallback: three file inputs
function FallbackLoader({ onLoad }: Props) {
  const [outputs, setOutputs] = useState<MetricSample[] | null>(null);
  const [gpu, setGpu] = useState<GpuSample[]>([]);
  const [engine, setEngine] = useState<EngineSample[]>([]);
  const [summary, setSummary] = useState<string | null>(null);
  const [frameHandles] = useState(new Map<string, FileSystemFileHandle>());
  const [error, setError] = useState('');

  async function handleOutputs(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text()) as MetricSample[];
      setOutputs(data);
    } catch { setError('Invalid outputs.json'); }
  }

  async function handleGpu(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try { setGpu(JSON.parse(await file.text()) as GpuSample[]); } catch { /* ignore */ }
  }

  async function handleEngine(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try { setEngine(JSON.parse(await file.text()) as EngineSample[]); } catch { /* ignore */ }
  }

  async function handleSummary(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setSummary(await file.text());
  }

  function submit() {
    if (!outputs) { setError('outputs.json is required'); return; }
    const runId = outputs[0]?.run_id ?? 'unknown';
    onLoad({ runId, outputs, gpu, engine, summary, frameHandles });
  }

  return (
    <div className="space-y-4">
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">outputs.json (required)</label>
        <input type="file" accept=".json" onChange={handleOutputs} className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" />
      </div>
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">gpu.json (optional)</label>
        <input type="file" accept=".json" onChange={handleGpu} className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" />
      </div>
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">engine.json (optional)</label>
        <input type="file" accept=".json" onChange={handleEngine} className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" />
      </div>
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-700">summary.txt (optional)</label>
        <input type="file" accept=".txt" onChange={handleSummary} className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" />
      </div>
      <button
        onClick={submit}
        disabled={!outputs}
        className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Load Run
      </button>
    </div>
  );
}

export function RunLoader({ onLoad }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showFallback, setShowFallback] = useState(false);
  const hasDirPicker = 'showDirectoryPicker' in window;
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
      <div className="bg-white rounded-2xl shadow-lg p-8 w-full max-w-md">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Verkos Simulator Dashboard</h1>
          <p className="text-gray-500 mt-1 text-sm">Open a run folder to explore results</p>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">{error}</div>
        )}

        {!showFallback ? (
          <div className="space-y-4">
            {hasDirPicker ? (
              <button
                onClick={() => loadViaDirectoryPicker(onLoad, setError, setLoading)}
                disabled={loading}
                className="w-full py-3 px-4 bg-blue-600 text-white rounded-xl font-medium text-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <span className="inline-block w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                  </svg>
                )}
                {loading ? 'Loading…' : 'Open Run Folder'}
              </button>
            ) : null}
            <button
              onClick={() => setShowFallback(true)}
              className="w-full py-2 px-4 border border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50 text-sm"
            >
              {hasDirPicker ? 'Use file inputs instead (Firefox)' : 'Select files manually'}
            </button>
          </div>
        ) : (
          <div>
            <button onClick={() => setShowFallback(false)} className="text-blue-600 text-sm mb-4 hover:underline">
              ← Back
            </button>
            <FallbackLoader onLoad={onLoad} />
          </div>
        )}
        <input ref={inputRef} type="file" className="hidden" />
      </div>
    </div>
  );
}
