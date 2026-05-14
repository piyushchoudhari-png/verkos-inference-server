import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { GpuSample } from '../types';

interface Props {
  gpu: GpuSample[];
}

const GPU_COLORS = ['#3b82f6', '#10b981', '#f97316', '#a855f7'];

export function GpuChart({ gpu }: Props) {
  if (gpu.length === 0) return null;

  const runStart = Math.min(...gpu.map(s => s.t_epoch));
  const gpuIndices = [...new Set(gpu.map(s => s.gpu_index))].sort((a, b) => a - b);

  // Build time-indexed data: one row per unique timestamp
  const timestamps = [...new Set(gpu.map(s => Math.round((s.t_epoch - runStart) * 10) / 10))].sort((a, b) => a - b);
  const data = timestamps.map(t => {
    const row: Record<string, number> = { t };
    for (const idx of gpuIndices) {
      const sample = gpu.find(s => Math.abs(s.t_epoch - runStart - t) < 0.1 && s.gpu_index === idx);
      if (sample) {
        row[`vram_${idx}`] = parseFloat((sample.memory_used_bytes / 1024 ** 3).toFixed(2));
        row[`util_${idx}`] = sample.gpu_util_pct;
      }
    }
    return row;
  });

  return (
    <section className="mb-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-3">GPU Metrics</h2>
      <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-6">
        <div>
          <p className="text-sm font-medium text-gray-600 mb-2">VRAM Used (GB)</p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="t" tickFormatter={v => `${v}s`} label={{ value: 'Time (s)', position: 'insideBottom', offset: -10 }} />
              <YAxis label={{ value: 'GB', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(v) => [`${v} GB`]} labelFormatter={l => `t=${l}s`} />
              <Legend />
              {gpuIndices.map((idx, i) => (
                <Line
                  key={idx}
                  type="monotone"
                  dataKey={`vram_${idx}`}
                  name={`GPU ${idx} VRAM`}
                  stroke={GPU_COLORS[i % GPU_COLORS.length]}
                  dot={false}
                  strokeWidth={1.5}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <p className="text-sm font-medium text-gray-600 mb-2">GPU Utilization (%)</p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="t" tickFormatter={v => `${v}s`} label={{ value: 'Time (s)', position: 'insideBottom', offset: -10 }} />
              <YAxis domain={[0, 100]} label={{ value: '%', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(v) => [`${v}%`]} labelFormatter={l => `t=${l}s`} />
              <Legend />
              {gpuIndices.map((idx, i) => (
                <Line
                  key={idx}
                  type="monotone"
                  dataKey={`util_${idx}`}
                  name={`GPU ${idx} util`}
                  stroke={GPU_COLORS[i % GPU_COLORS.length]}
                  dot={false}
                  strokeWidth={1.5}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
