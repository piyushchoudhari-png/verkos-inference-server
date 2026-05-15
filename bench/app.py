from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Project root on path so both `src.*` and `bench.*` are importable
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# Clean up temp configs older than 24 h
_tmp = ROOT / "bench" / "tmp"
if _tmp.exists():
    cutoff = time.time() - 86400
    for f in _tmp.glob("*.yaml"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)

st.set_page_config(
    page_title="Verkos Test Bench",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Session state keys
for _key in ("selected_run_id", "last_run_id", "active_subprocess", "video_paths"):
    if _key not in st.session_state:
        st.session_state[_key] = None

st.title("Verkos Inference Test Bench")

tab_results, tab_configure = st.tabs(["🔬 Results", "⚙️ Configure & Run"])

with tab_results:
    from bench.tabs.results import render_results
    render_results(RUNS_DIR)

with tab_configure:
    from bench.tabs.configure import render_configure
    render_configure(ROOT, RUNS_DIR)
