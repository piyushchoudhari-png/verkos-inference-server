import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { MetricSample, AggregatedStats } from '../types';

interface Props {
  outputs: MetricSample[];
  stats: AggregatedStats;
}

export function ErrorPanel({ outputs, stats }: Props) {
  if (stats.error_count === 0) return null;

  const errors = outputs.filter(s => s.status === 'error');
  const barData = Object.entries(stats.error_classes).map(([cls, cnt]) => ({ cls, cnt }));

  return (
    <section className="mb-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-3">Errors ({stats.error_count})</h2>
      <div className="bg-white rounded-xl border border-red-200 overflow-hidden">
        {barData.length > 0 && (
          <div className="p-4 border-b border-gray-100">
            <p className="text-sm font-medium text-gray-600 mb-2">By class</p>
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={barData} layout="vertical" margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis type="category" dataKey="cls" width={60} />
                <Tooltip />
                <Bar dataKey="cnt" name="Count" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-red-50 text-red-700 text-xs uppercase">
              <tr>
                <th className="px-4 py-2 text-left">Feed</th>
                <th className="px-4 py-2 text-left">Frame</th>
                <th className="px-4 py-2 text-left">Class</th>
                <th className="px-4 py-2 text-left">Message</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {errors.map(s => (
                <tr key={`${s.feed_id}-${s.frame_index}`} className="hover:bg-red-50">
                  <td className="px-4 py-2 font-mono">{s.feed_id}</td>
                  <td className="px-4 py-2 font-mono">{s.frame_index}</td>
                  <td className="px-4 py-2">
                    <span className="px-2 py-0.5 bg-red-100 text-red-700 rounded text-xs">
                      {classifyError(s.error_msg)}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-gray-600 text-xs max-w-xs truncate">{s.error_msg ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function classifyError(msg: string | null): string {
  if (!msg) return 'other';
  const m = msg.toLowerCase();
  if (m.includes('out of memory') || m.includes('oom')) return 'oom';
  if (m.includes('timeout') || m.includes('timed out')) return 'timeout';
  if (m.includes('cuda') || m.includes('nccl')) return 'cuda';
  if (m.includes('tokeniz') || m.includes('vocab') || m.includes('model')) return 'model';
  return 'other';
}
