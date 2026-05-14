import { useState } from 'react';
import type { MetricSample } from '../types';
import { parseDetections } from '../lib/parseDetections';
import { BBoxCanvas } from './BBoxCanvas';

interface Props {
  sample: MetricSample;
  frameHandle: FileSystemFileHandle | undefined;
}

function ms(s: number): string {
  const v = s * 1000;
  if (v >= 1000) return `${(v / 1000).toFixed(2)}s`;
  return `${Math.round(v)}ms`;
}

export function FrameCard({ sample, frameHandle }: Props) {
  const [expanded, setExpanded] = useState(false);
  const parsed = parseDetections(sample.output_text);
  const isError = sample.status === 'error';

  const frameKey = `feed${String(sample.feed_id).padStart(2, '0')}_frame${String(sample.frame_index).padStart(4, '0')}`;

  return (
    <div
      className={`rounded-xl border bg-white overflow-hidden break-inside-avoid ${isError ? 'border-red-300' : 'border-gray-200'}`}
    >
      {!isError && (
        <BBoxCanvas
          frameHandle={frameHandle}
          detections={parsed.detections}
          resolutionW={sample.resolution_w}
          resolutionH={sample.resolution_h}
        />
      )}
      {isError && (
        <div className="bg-red-50 p-2 text-red-600 text-xs border-b border-red-200">
          {sample.error_msg ?? 'error'}
        </div>
      )}

      <div className="p-3 space-y-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-xs text-gray-500">{frameKey}</span>
          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${isError ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
            {sample.status}
          </span>
          <span className="text-xs text-gray-500 ml-auto">{ms(sample.latency_s)}</span>
          {sample.ttft_s != null && (
            <span className="text-xs text-gray-400">TTFT {ms(sample.ttft_s)}</span>
          )}
        </div>

        <div className="flex gap-3 text-xs text-gray-400">
          <span>{sample.prompt_tokens}↑ {sample.completion_tokens}↓ tok</span>
          <span>{sample.resolution_w}×{sample.resolution_h}</span>
        </div>

        {parsed.detections.length > 0 && (
          <ul className="text-xs space-y-0.5">
            {parsed.detections.map((d, i) => (
              <li key={i} className="flex gap-2">
                <span className="font-medium text-gray-700">{d.event}</span>
                <span className="text-gray-400">×{d.count}</span>
                {d.confidence != null && <span className="text-gray-400">{(d.confidence * 100).toFixed(0)}%</span>}
              </li>
            ))}
          </ul>
        )}

        {parsed.rawError && (
          <p className="text-xs text-amber-600">Parse error: {parsed.rawError}</p>
        )}

        <button
          onClick={() => setExpanded(v => !v)}
          className="text-xs text-blue-500 hover:underline"
        >
          {expanded ? 'Hide' : 'Show'} raw output
        </button>

        {expanded && (
          <pre className="text-xs bg-gray-50 border border-gray-200 rounded p-2 overflow-x-auto whitespace-pre-wrap break-words max-h-48">
            {sample.output_text ?? '(empty)'}
          </pre>
        )}
      </div>
    </div>
  );
}
