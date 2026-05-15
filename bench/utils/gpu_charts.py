from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def _t_series(samples: list[dict[str, Any]], key: str = "t_epoch") -> tuple[float, list[float]]:
    epochs = [s[key] for s in samples]
    t0 = min(epochs) if epochs else 0.0
    return t0, [e - t0 for e in epochs]


def vram_chart(
    gpu_samples: list[dict[str, Any]],
    idle_baseline: dict[int, int] | None = None,
) -> go.Figure:
    fig = go.Figure()
    gpu_indices = sorted({s["gpu_index"] for s in gpu_samples})
    for gi in gpu_indices:
        subs = sorted(
            [s for s in gpu_samples if s["gpu_index"] == gi],
            key=lambda s: s["t_epoch"],
        )
        t0, x = _t_series(subs)
        y = [s["memory_used_bytes"] / 1e9 for s in subs]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"GPU {gi}"))
        if idle_baseline and gi in idle_baseline:
            idle_gb = idle_baseline[gi] / 1e9
            fig.add_hline(
                y=idle_gb,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"GPU {gi} idle",
                annotation_position="bottom right",
            )
    fig.update_layout(
        title="VRAM Usage",
        xaxis_title="Time (s)",
        yaxis_title="Memory Used (GB)",
        margin=dict(t=40, b=40, l=60, r=20),
        height=300,
    )
    return fig


def gpu_util_chart(gpu_samples: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    for gi in sorted({s["gpu_index"] for s in gpu_samples}):
        subs = sorted(
            [s for s in gpu_samples if s["gpu_index"] == gi],
            key=lambda s: s["t_epoch"],
        )
        _, x = _t_series(subs)
        fig.add_trace(go.Scatter(x=x, y=[s["gpu_util_pct"] for s in subs], mode="lines", name=f"GPU {gi}"))
    fig.update_layout(
        title="GPU Utilization",
        xaxis_title="Time (s)",
        yaxis_title="Utilization (%)",
        yaxis=dict(range=[0, 100]),
        margin=dict(t=40, b=40, l=60, r=20),
        height=300,
    )
    return fig


def power_chart(gpu_samples: list[dict[str, Any]]) -> go.Figure | None:
    if all(s.get("power_w") is None for s in gpu_samples):
        return None
    fig = go.Figure()
    for gi in sorted({s["gpu_index"] for s in gpu_samples}):
        subs = sorted(
            [s for s in gpu_samples if s["gpu_index"] == gi and s.get("power_w") is not None],
            key=lambda s: s["t_epoch"],
        )
        if not subs:
            continue
        _, x = _t_series(subs)
        fig.add_trace(go.Scatter(x=x, y=[s["power_w"] for s in subs], mode="lines", name=f"GPU {gi}"))
    fig.update_layout(
        title="Power Draw",
        xaxis_title="Time (s)",
        yaxis_title="Power (W)",
        margin=dict(t=40, b=40, l=60, r=20),
        height=300,
    )
    return fig


def temp_chart(gpu_samples: list[dict[str, Any]]) -> go.Figure | None:
    if all(s.get("temperature_c") is None for s in gpu_samples):
        return None
    fig = go.Figure()
    for gi in sorted({s["gpu_index"] for s in gpu_samples}):
        subs = sorted(
            [s for s in gpu_samples if s["gpu_index"] == gi and s.get("temperature_c") is not None],
            key=lambda s: s["t_epoch"],
        )
        if not subs:
            continue
        _, x = _t_series(subs)
        fig.add_trace(go.Scatter(x=x, y=[s["temperature_c"] for s in subs], mode="lines", name=f"GPU {gi}"))
    fig.update_layout(
        title="Temperature",
        xaxis_title="Time (s)",
        yaxis_title="Temperature (°C)",
        margin=dict(t=40, b=40, l=60, r=20),
        height=300,
    )
    return fig


def engine_chart(engine_samples: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if not engine_samples:
        return fig
    samps = sorted(engine_samples, key=lambda s: s["t_epoch"])
    _, x = _t_series(samps)

    for name, color in [("num_running", "#1f77b4"), ("num_waiting", "#ff7f0e"), ("num_swapped", "#2ca02c")]:
        fig.add_trace(go.Scatter(x=x, y=[s[name] for s in samps], mode="lines", name=name, line=dict(color=color)))

    fig.add_trace(go.Scatter(
        x=x,
        y=[s["gpu_kv_cache_usage_pct"] for s in samps],
        mode="lines",
        name="KV cache %",
        yaxis="y2",
        line=dict(dash="dot", color="#9467bd"),
    ))

    fig.update_layout(
        title="vLLM Scheduler Stats",
        xaxis_title="Time (s)",
        yaxis=dict(title="Requests"),
        yaxis2=dict(title="KV Cache %", overlaying="y", side="right", range=[0, 100]),
        margin=dict(t=40, b=40, l=60, r=60),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
