import {
  ComposedChart,
  Scatter,
  ReferenceLine,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { MetricSample, AggregatedStats } from '../types';

interface Props {
  outputs: MetricSample[];
  stats: AggregatedStats;
}

const FEED_COLORS = ['#3b82f6', '#f97316', '#10b981', '#a855f7', '#ef4444', '#06b6d4'];

export function LatencyChart({ outputs, stats }: Props) {
  const runStart = outputs.length > 0 ? Math.min(...outputs.map(s => s.t0_epoch)) : 0;

  const feedIds = [...new Set(outputs.map(s => s.feed_id))].sort((a, b) => a - b);

  const dataByFeed = feedIds.map((fid, i) => ({
    feedId: fid,
    color: FEED_COLORS[i % FEED_COLORS.length],
    points: outputs
      .filter(s => s.feed_id === fid && s.status === 'ok')
      .map(s => ({
        t: parseFloat((s.t0_epoch - runStart).toFixed(2)),
        lat: parseFloat((s.latency_s * 1000).toFixed(1)),
        frame: s.frame_index,
        feed: s.feed_id,
      })),
  }));

  return (
    <section className="mb-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-3">Latency Timeline</h2>
      <div className="bg-white rounded-xl border border-gray-200 p-4">
        <ResponsiveContainer width="100%" height={280}>
          <ComposedChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="t"
              type="number"
              domain={['auto', 'auto']}
              label={{ value: 'Time (s)', position: 'insideBottom', offset: -10 }}
              tickFormatter={v => `${v}s`}
            />
            <YAxis
              dataKey="lat"
              label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', offset: 10 }}
              tickFormatter={v => `${v}`}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload as { t: number; lat: number; frame: number; feed: number };
                return (
                  <div className="bg-white border border-gray-200 rounded shadow p-2 text-xs">
                    <p>Feed {d.feed} · Frame {d.frame}</p>
                    <p>t = {d.t}s</p>
                    <p>Latency: {d.lat}ms</p>
                  </div>
                );
              }}
            />
            <Legend />
            <ReferenceLine
              y={stats.latency_p50_ms}
              stroke="#3b82f6"
              strokeDasharray="6 3"
              label={{ value: `P50 ${stats.latency_p50_ms}ms`, position: 'right', fontSize: 11 }}
            />
            <ReferenceLine
              y={stats.latency_p95_ms}
              stroke="#ef4444"
              strokeDasharray="6 3"
              label={{ value: `P95 ${stats.latency_p95_ms}ms`, position: 'right', fontSize: 11 }}
            />
            {dataByFeed.map(({ feedId, color, points }) => (
              <Scatter
                key={feedId}
                name={`Feed ${feedId}`}
                data={points}
                fill={color}
                opacity={0.7}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
