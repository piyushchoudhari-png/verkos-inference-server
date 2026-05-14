import { useState } from 'react';
import type { RunData } from './types';
import { aggregate } from './lib/aggregate';
import { RunLoader } from './components/RunLoader';
import { RunOverview } from './components/RunOverview';
import { FrameExplorer } from './components/FrameExplorer';
import { LatencyChart } from './components/LatencyChart';
import { GpuChart } from './components/GpuChart';
import { EngineChart } from './components/EngineChart';
import { ErrorPanel } from './components/ErrorPanel';
import { ExportButton } from './components/ExportButton';

export default function App() {
  const [runData, setRunData] = useState<RunData | null>(null);

  if (!runData) {
    return <RunLoader onLoad={setRunData} />;
  }

  const stats = aggregate(runData.outputs);

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="sticky top-0 z-10 bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-gray-900">Verkos Simulator</h1>
            <p className="text-xs text-gray-500 font-mono">{runData.runId}</p>
          </div>
          <div className="flex items-center gap-3">
            <ExportButton runId={runData.runId} />
            <button
              onClick={() => setRunData(null)}
              className="text-sm text-gray-500 hover:text-gray-800 border border-gray-300 px-3 py-1.5 rounded-lg"
            >
              Close run
            </button>
          </div>
        </div>
      </header>

      <main id="dashboard-root" className="max-w-screen-2xl mx-auto px-6 py-6">
        <RunOverview stats={stats} outputs={runData.outputs} />
        <LatencyChart outputs={runData.outputs} stats={stats} />
        <GpuChart gpu={runData.gpu} />
        <EngineChart engine={runData.engine} />
        <ErrorPanel outputs={runData.outputs} stats={stats} />
        <FrameExplorer outputs={runData.outputs} frameHandles={runData.frameHandles} />
      </main>
    </div>
  );
}
