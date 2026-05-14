import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { EngineSample } from '../types';

interface Props {
  engine: EngineSample[];
}

export function EngineChart({ engine }: Props) {
  if (engine.length === 0) return null;

  const runStart = Math.min(...engine.map(s => s.t_epoch));
  const data = engine.map(s => ({
    t: parseFloat((s.t_epoch - runStart).toFixed(1)),
    running: s.num_running,
    waiting: s.num_waiting,
    swapped: s.num_swapped,
    kv: parseFloat(s.gpu_kv_cache_usage_pct.toFixed(1)),
  })).sort((a, b) => a.t - b.t);

  return (
    <section className="mb-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-3">Engine Metrics</h2>
      <div className="bg-white rounded-xl border border-gray-200 p-4 space-y-6">
        <div>
          <p className="text-sm font-medium text-gray-600 mb-2">Scheduler Queue</p>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="t" tickFormatter={v => `${v}s`} label={{ value: 'Time (s)', position: 'insideBottom', offset: -10 }} />
              <YAxis allowDecimals={false} />
              <Tooltip labelFormatter={l => `t=${l}s`} />
              <Legend />
              <Area type="monotone" dataKey="running" name="Running" stackId="1" stroke="#3b82f6" fill="#3b82f620" strokeWidth={1.5} />
              <Area type="monotone" dataKey="waiting" name="Waiting" stackId="1" stroke="#f97316" fill="#f9731620" strokeWidth={1.5} />
              <Area type="monotone" dataKey="swapped" name="Swapped" stackId="1" stroke="#a855f7" fill="#a855f720" strokeWidth={1.5} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div>
          <p className="text-sm font-medium text-gray-600 mb-2">GPU KV-Cache Usage (%)</p>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="t" tickFormatter={v => `${v}s`} label={{ value: 'Time (s)', position: 'insideBottom', offset: -10 }} />
              <YAxis domain={[0, 100]} label={{ value: '%', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(v) => [`${v}%`]} labelFormatter={l => `t=${l}s`} />
              <Line type="monotone" dataKey="kv" name="KV cache" stroke="#10b981" dot={false} strokeWidth={1.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
