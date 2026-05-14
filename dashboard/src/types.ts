export interface MetricSample {
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

export interface GpuSample {
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

export interface EngineSample {
  run_id: string;
  t_epoch: number;
  num_running: number;
  num_waiting: number;
  num_swapped: number;
  gpu_kv_cache_usage_pct: number;
  cpu_kv_cache_usage_pct: number;
  num_preemption_total: number;
}

export interface RunData {
  runId: string;
  outputs: MetricSample[];
  gpu: GpuSample[];
  engine: EngineSample[];
  summary: string | null;
  frameHandles: Map<string, FileSystemFileHandle>;
}

export interface Detection {
  event: string;
  count: number;
  boxes: [number, number, number, number][];
  confidence?: number;
}

export interface ParsedOutput {
  detections: Detection[];
  notes?: string;
  rawError?: string;
}

export interface AggregatedStats {
  total_frames: number;
  successful_frames: number;
  error_count: number;
  error_rate_pct: number;
  error_classes: Record<string, number>;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  latency_mean_ms: number;
  throughput_fps: number;
  run_duration_s: number;
  ttft_p50_ms: number;
  per_feed: Record<number, { frames: number; latency_p50_ms: number; latency_p95_ms: number }>;
}
