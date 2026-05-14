import type { AggregatedStats, MetricSample } from '../types';

interface Props {
  stats: AggregatedStats;
  outputs: MetricSample[];
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  );
}

function ms(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(1)}s`;
  return `${Math.round(v)}ms`;
}

export function RunOverview({ stats, outputs }: Props) {
  const model = outputs[0]?.model ?? '—';
  const mode = outputs[0]?.inference_mode ?? '—';

  return (
    <section className="mb-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-3">Overview</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <StatCard label="Total Frames" value={String(stats.total_frames)} />
        <StatCard
          label="Success / Error"
          value={`${stats.successful_frames} / ${stats.error_count}`}
          sub={`${stats.error_rate_pct}% error rate`}
        />
        <StatCard
          label="Latency P50 / P95"
          value={`${ms(stats.latency_p50_ms)} / ${ms(stats.latency_p95_ms)}`}
          sub={`P99 ${ms(stats.latency_p99_ms)} · mean ${ms(stats.latency_mean_ms)}`}
        />
        <StatCard
          label="TTFT P50"
          value={stats.ttft_p50_ms > 0 ? ms(stats.ttft_p50_ms) : '—'}
        />
        <StatCard
          label="Throughput"
          value={`${stats.throughput_fps.toFixed(3)} fps`}
          sub={`${stats.run_duration_s}s run`}
        />
        <StatCard label="Model" value={model.split('/').pop() ?? model} sub={mode} />
      </div>
    </section>
  );
}
