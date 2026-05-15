from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def list_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Return all run_meta.json records from runs_dir, newest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    runs: list[dict[str, Any]] = []
    for meta_path in runs_dir.glob("*/run_meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
            meta["_run_dir"] = str(meta_path.parent)
            runs.append(meta)
        except Exception:
            continue
    runs.sort(key=lambda r: r.get("started_at_iso", ""), reverse=True)
    return runs


def load_outputs(run_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "outputs.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def load_gpu(run_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "gpu.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def load_engine(run_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "engine.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def load_config_snapshot(run_dir: str | Path) -> dict[str, Any]:
    p = Path(run_dir) / "config_snapshot.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_baselines(run_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "baselines.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def frames_for_feed(run_dir: str | Path, feed_id: int) -> list[Path]:
    frames_dir = Path(run_dir) / "frames"
    if not frames_dir.exists():
        return []
    return sorted(frames_dir.glob(f"feed{feed_id:02d}_frame*.jpg"))
