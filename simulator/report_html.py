"""Self-contained Plotly HTML reports for runs and sweeps.

Each report bundles plotly.js inline so the file works offline / when emailed.
"""

import html as html_lib
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator.config import SimConfig
from simulator.metrics import EngineSample, GpuSample, IdleBaseline, MetricSample, aggregate

_HTML_SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #1f2937; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; margin-top: 32px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }}
  table.kv {{ border-collapse: collapse; margin: 12px 0; }}
  table.kv td {{ padding: 2px 12px 2px 0; font-size: 13px; }}
  table.kv td.k {{ color: #6b7280; }}
  table.data {{ border-collapse: collapse; font-size: 13px; }}
  table.data th, table.data td {{ border: 1px solid #e5e7eb; padding: 4px 8px; text-align: right; }}
  table.data th {{ background: #f9fafb; text-align: center; }}
  details {{ margin: 8px 0; }}
  summary {{ cursor: pointer; color: #2563eb; font-size: 13px; }}
  pre {{ background: #f9fafb; border: 1px solid #e5e7eb; padding: 8px; font-size: 12px;
        white-space: pre-wrap; word-break: break-word; }}
  .meta {{ color: #6b7280; font-size: 12px; margin-bottom: 16px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _kv_table(rows: list[tuple[str, Any]]) -> str:
    parts = ['<table class="kv">']
    for k, v in rows:
        parts.append(f'<tr><td class="k">{html_lib.escape(str(k))}</td><td>{html_lib.escape(str(v))}</td></tr>')
    parts.append("</table>")
    return "\n".join(parts)


def _fig_html(fig: go.Figure, include_plotlyjs: bool) -> str:
    html: str = fig.to_html(
        include_plotlyjs="inline" if include_plotlyjs else False,
        full_html=False,
        config={"displaylogo": False},
    )
    return html


def _elapsed(ts: list[float], t0: float) -> list[float]:
    return [t - t0 for t in ts]


def _peak_vram_gb(gpu_samples: list[GpuSample]) -> float:
    if not gpu_samples:
        return 0.0
    return max(s.memory_used_bytes for s in gpu_samples) / 1024**3


def _rolling_p95(samples: list[MetricSample], window_s: float = 10.0) -> tuple[list[float], list[float]]:
    ok = sorted([s for s in samples if s.status == "ok"], key=lambda s: s.t0_epoch)
    if not ok:
        return [], []
    t0 = ok[0].t0_epoch
    xs: list[float] = []
    ys: list[float] = []
    for i, s in enumerate(ok):
        lo = s.t0_epoch - window_s
        window = [w.latency_s * 1000.0 for w in ok[: i + 1] if w.t0_epoch >= lo]
        window.sort()
        idx = int(0.95 * (len(window) - 1))
        xs.append(s.t0_epoch - t0)
        ys.append(window[idx])
    return xs, ys


def _resource_fig(gpu_samples: list[GpuSample], engine_samples: list[EngineSample], t0: float) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("VRAM used (GB)", "KV cache % + queue depth", "GPU util % / power (W) / temp (°C)"),
    )

    # GPU 0 only for simplicity in the resource panel (multi-GPU shown separately if needed)
    g0 = [s for s in gpu_samples if s.gpu_index == 0]
    if g0:
        xs = _elapsed([s.t_epoch for s in g0], t0)
        fig.add_trace(
            go.Scatter(x=xs, y=[s.memory_used_bytes / 1024**3 for s in g0], name="VRAM used", mode="lines"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[s.torch_allocated_bytes / 1024**3 for s in g0],
                name="torch alloc",
                mode="lines",
                line={"dash": "dot"},
            ),
            row=1,
            col=1,
        )

    if engine_samples:
        xs = _elapsed([s.t_epoch for s in engine_samples], t0)
        fig.add_trace(
            go.Scatter(x=xs, y=[s.gpu_kv_cache_usage_pct for s in engine_samples], name="KV cache %", mode="lines"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=xs, y=[s.num_running for s in engine_samples], name="running", mode="lines"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=xs, y=[s.num_waiting for s in engine_samples], name="waiting", mode="lines"),
            row=2,
            col=1,
        )

    if g0:
        xs = _elapsed([s.t_epoch for s in g0], t0)
        fig.add_trace(go.Scatter(x=xs, y=[s.gpu_util_pct for s in g0], name="util %", mode="lines"), row=3, col=1)
        if any(s.power_w is not None for s in g0):
            fig.add_trace(
                go.Scatter(x=xs, y=[s.power_w for s in g0], name="power W", mode="lines"),
                row=3,
                col=1,
            )
        if any(s.temperature_c is not None for s in g0):
            fig.add_trace(
                go.Scatter(x=xs, y=[s.temperature_c for s in g0], name="temp °C", mode="lines"),
                row=3,
                col=1,
            )

    fig.update_xaxes(title_text="elapsed (s)", row=3, col=1)
    fig.update_layout(height=720, margin={"l": 50, "r": 20, "t": 40, "b": 40})
    return fig


def _latency_over_time_fig(samples: list[MetricSample]) -> go.Figure:
    ok = sorted([s for s in samples if s.status == "ok"], key=lambda s: s.t0_epoch)
    if not ok:
        return go.Figure()
    t0 = ok[0].t0_epoch
    xs = [s.t0_epoch - t0 for s in ok]
    ys = [s.latency_s * 1000.0 for s in ok]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            name="per-request latency",
            mode="markers",
            marker={"size": 5, "opacity": 0.5},
            text=[f"feed {s.feed_id} frame {s.frame_index}" for s in ok],
        )
    )
    rxs, rys = _rolling_p95(samples)
    if rxs:
        fig.add_trace(go.Scatter(x=rxs, y=rys, name="rolling P95 (10s)", mode="lines"))
    fig.update_layout(
        title="Latency over time",
        xaxis_title="elapsed (s)",
        yaxis_title="latency (ms)",
        height=360,
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
    )
    return fig


def _cdf_fig(latencies_ms: list[float]) -> go.Figure:
    fig = go.Figure()
    if not latencies_ms:
        return fig
    n = len(latencies_ms)
    ys = [(i + 1) / n * 100.0 for i in range(n)]
    fig.add_trace(go.Scatter(x=latencies_ms, y=ys, mode="lines", name="CDF"))
    for p in (50, 95, 99):
        idx = max(0, int(p / 100 * (n - 1)))
        fig.add_vline(x=latencies_ms[idx], line_dash="dot", annotation_text=f"P{p}")
    fig.update_layout(
        title="Latency CDF",
        xaxis_title="latency (ms)",
        yaxis_title="cumulative %",
        height=360,
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
    )
    return fig


def _per_feed_box_fig(samples: list[MetricSample]) -> go.Figure:
    ok = [s for s in samples if s.status == "ok"]
    fig = go.Figure()
    if not ok:
        return fig
    for fid in sorted({s.feed_id for s in ok}):
        ys = [s.latency_s * 1000.0 for s in ok if s.feed_id == fid]
        fig.add_trace(go.Box(y=ys, name=f"feed {fid}", boxmean=True))
    fig.update_layout(
        title="Per-feed latency distribution",
        yaxis_title="latency (ms)",
        height=360,
        margin={"l": 50, "r": 20, "t": 40, "b": 40},
    )
    return fig


def _headline_table(
    stats: dict[str, Any],
    config: SimConfig,
    baselines: list[IdleBaseline],
    gpu_samples: list[GpuSample],
    run_id: str,
) -> str:
    peak_vram = _peak_vram_gb(gpu_samples)
    idle_vram = baselines[0].memory_used_bytes / 1024**3 if baselines else 0.0
    rows: list[tuple[str, Any]] = [
        ("run_id", run_id),
        ("model", config.model_path),
        ("num_feeds", config.num_feeds),
        ("frame_interval_s", config.frame_interval_s),
        ("resolution", config.resolution or f"native (downscale={config.downscale_factor})"),
        ("max_tokens", config.max_tokens),
        ("frames ok / total", f"{stats.get('successful_frames', 0)} / {stats.get('total_frames', 0)}"),
        ("error rate", f"{stats.get('error_rate_pct', 0)}%"),
        ("latency P50 / P95 / P99 (ms)",
         f"{stats.get('latency_p50_ms', 0)} / {stats.get('latency_p95_ms', 0)} / {stats.get('latency_p99_ms', 0)}"),
        ("throughput (fps)", stats.get("throughput_fps", 0)),
        ("throughput CoV (10s)", stats.get("throughput_cov_10s", 0)),
        ("prefill tok/s P50 / P95", f"{stats.get('prefill_tps_p50', 0)} / {stats.get('prefill_tps_p95', 0)}"),
        ("decode tok/s P50 / P95", f"{stats.get('decode_tps_p50', 0)} / {stats.get('decode_tps_p95', 0)}"),
        ("idle VRAM (GB)", round(idle_vram, 2)),
        ("peak VRAM (GB)", round(peak_vram, 2)),
        ("suggested min VRAM spec (peak × 1.2 GB)", round(peak_vram * 1.2, 2)),
        ("run duration (s)", stats.get("run_duration_s", 0)),
    ]
    return _kv_table(rows)


def _error_table(stats: dict[str, Any]) -> str:
    classes: dict[str, int] = stats.get("error_classes", {}) or {}
    if not classes:
        return "<p class='meta'>No errors.</p>"
    parts = ["<table class='data'><tr><th>class</th><th>count</th></tr>"]
    for cls, cnt in sorted(classes.items(), key=lambda x: -x[1]):
        parts.append(f"<tr><td style='text-align:left'>{html_lib.escape(cls)}</td><td>{cnt}</td></tr>")
    parts.append("</table>")
    return "\n".join(parts)


def _sample_outputs(samples: list[MetricSample], limit: int = 20) -> str:
    ok = [s for s in samples if s.status == "ok" and s.output_text]
    if not ok:
        return "<p class='meta'>No outputs to display.</p>"
    parts = []
    for s in ok[:limit]:
        title = f"feed {s.feed_id} · frame {s.frame_index} · {s.latency_s * 1000:.0f} ms · {s.completion_tokens} tok"
        text = html_lib.escape(s.output_text or "")
        parts.append(f"<details><summary>{html_lib.escape(title)}</summary><pre>{text}</pre></details>")
    return "\n".join(parts)


def write_run_report(
    metric_samples: list[MetricSample],
    gpu_samples: list[GpuSample],
    engine_samples: list[EngineSample],
    baselines: list[IdleBaseline],
    config: SimConfig,
    run_id: str,
    output_path: Path,
) -> Path:
    stats = aggregate(metric_samples)

    t0 = min(
        (s.t0_epoch for s in metric_samples),
        default=min((s.t_epoch for s in gpu_samples), default=0.0),
    )

    figs = [
        ("Resources", _resource_fig(gpu_samples, engine_samples, t0)),
        ("Latency over time", _latency_over_time_fig(metric_samples)),
        ("Latency CDF", _cdf_fig(stats.get("latency_ms_sorted", []) or [])),
        ("Per-feed latency", _per_feed_box_fig(metric_samples)),
    ]

    body_parts = [
        f"<h1>Sim Run: {html_lib.escape(run_id)}</h1>",
        f"<div class='meta'>Model: {html_lib.escape(config.model_path)}</div>",
        "<h2>Headline</h2>",
        _headline_table(stats, config, baselines, gpu_samples, run_id),
    ]
    first = True
    for title, fig in figs:
        body_parts.append(f"<h2>{html_lib.escape(title)}</h2>")
        body_parts.append(_fig_html(fig, include_plotlyjs=first))
        first = False

    body_parts += [
        "<h2>Errors</h2>",
        _error_table(stats),
        "<h2>Sample outputs (first 20)</h2>",
        _sample_outputs(metric_samples, limit=20),
    ]

    html = _HTML_SHELL.format(title=f"Sim Run {run_id}", body="\n".join(body_parts))
    output_path.write_text(html)
    return output_path


def _heatmap_fig(
    title: str,
    x_axis: tuple[str, list[Any]],
    y_axis: tuple[str, list[Any]],
    z: list[list[float | None]],
    colorscale: str = "Viridis",
    reverse: bool = False,
) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            x=[str(v) for v in x_axis[1]],
            y=[str(v) for v in y_axis[1]],
            z=z,
            colorscale=colorscale,
            reversescale=reverse,
            hovertemplate=f"{x_axis[0]}=%{{x}}<br>{y_axis[0]}=%{{y}}<br>value=%{{z}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_axis[0],
        yaxis_title=y_axis[0],
        height=400,
        margin={"l": 80, "r": 20, "t": 40, "b": 40},
    )
    return fig


def write_sweep_report(
    sweep_id: str,
    axes: dict[str, list[Any]],
    runs: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Render heatmaps across the first two sweep axes.

    Each entry in `runs` must have axis values under their axis names plus the metric keys
    `latency_p95_ms`, `throughput_fps`, `peak_vram_gb`, `error_rate_pct`, `run_id`, `report_path`.
    """
    body_parts = [f"<h1>Sweep: {html_lib.escape(sweep_id)}</h1>"]

    axis_names = list(axes.keys())
    if not axis_names:
        body_parts.append("<p>No axes defined.</p>")
    elif len(axis_names) == 1:
        ax = axis_names[0]
        vals = axes[ax]
        for metric, friendly, _reverse in [
            ("latency_p95_ms", "P95 latency (ms)", False),
            ("throughput_fps", "Throughput (fps)", True),
            ("peak_vram_gb", "Peak VRAM (GB)", False),
            ("error_rate_pct", "Error rate (%)", False),
        ]:
            ys = [next((r[metric] for r in runs if r.get(ax) == v), None) for v in vals]
            fig = go.Figure(data=go.Bar(x=[str(v) for v in vals], y=ys))
            fig.update_layout(title=friendly, xaxis_title=ax, height=320)
            body_parts.append(f"<h2>{friendly}</h2>")
            body_parts.append(_fig_html(fig, include_plotlyjs=(metric == "latency_p95_ms")))
    else:
        x_name, y_name = axis_names[0], axis_names[1]
        x_vals, y_vals = axes[x_name], axes[y_name]

        first = True
        for metric, friendly, reverse in [
            ("latency_p95_ms", "P95 latency (ms)", False),
            ("throughput_fps", "Throughput (fps)", True),
            ("peak_vram_gb", "Peak VRAM (GB)", False),
            ("error_rate_pct", "Error rate (%)", False),
        ]:
            z: list[list[float | None]] = []
            for yv in y_vals:
                z_row: list[float | None] = []
                for xv in x_vals:
                    match = [
                        r for r in runs
                        if r.get(x_name) == xv and r.get(y_name) == yv
                    ]
                    z_row.append(match[0][metric] if match else None)
                z.append(z_row)
            fig = _heatmap_fig(friendly, (x_name, x_vals), (y_name, y_vals), z, reverse=reverse)
            body_parts.append(f"<h2>{friendly}</h2>")
            body_parts.append(_fig_html(fig, include_plotlyjs=first))
            first = False

    # Per-run links
    body_parts.append("<h2>Runs</h2>")
    if runs:
        headers = list(axes.keys()) + ["latency_p95_ms", "throughput_fps", "peak_vram_gb", "error_rate_pct", "report"]
        parts = ["<table class='data'><tr>" + "".join(f"<th>{html_lib.escape(h)}</th>" for h in headers) + "</tr>"]
        for r in runs:
            row = "<tr>"
            for h in headers[:-1]:
                row += f"<td>{html_lib.escape(str(r.get(h, '')))}</td>"
            link = r.get("report_path")
            if link:
                row += f"<td style='text-align:left'><a href='{html_lib.escape(str(link))}'>open</a></td>"
            else:
                row += "<td></td>"
            row += "</tr>"
            parts.append(row)
        parts.append("</table>")
        body_parts.append("\n".join(parts))

    html = _HTML_SHELL.format(title=f"Sweep {sweep_id}", body="\n".join(body_parts))
    output_path.write_text(html)
    return output_path
