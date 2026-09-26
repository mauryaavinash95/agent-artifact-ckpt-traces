"""
Normalized cross-plane (C0-C3) event schema for the unified-agentic-memory
characterization.

Design rule (extract-and-replay, NOT curve-fitting): extractors read a REAL
trace and emit normalized events. A composer places events on a shared clock.
Analyzers consume the normalized CSV and emit I/O trends. No distribution
fitting / sampling in this pipeline.

Event columns
-------------
ts            float   wall-clock seconds on the SHARED scenario clock (t=0 = scenario start)
scenario      str     scenario name (e.g. "coding_swarm")
plane         str     one of C0/C1/C2/C3 (see PLANES)
obj_type      str     fine-grained object type (kv, tool_call, artifact, mem_obs, ...)
op            str     "read" | "write"
logical_bytes int     logical payload size of the object
phys_bytes    int     bytes actually moved to/from storage for this op
                      (>= logical_bytes when a whole-file rewrite / backup is modeled)
agent_id      str     agent / worker identity within the scenario
session_id    str     source session identity (provenance)
req_id        str     source round/request identity (provenance)
src           str     source trace name (tracelab, mooncake, locomo, perltqa, static:<repo>)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict, fields
from typing import Iterable, List

# ---------------------------------------------------------------------------
# planes
# ---------------------------------------------------------------------------
PLANES = {
    "C0": "S_ctx: model context (KV)",
    "C1": "S_exec: execution state",
    "C2": "S_cogn: cognitive memory",
    "C3": "S_arti: bulk artifacts + embeddings",
}
# stable plotting order + label used across all figures
# (internal keys stay C0..C3; display labels use the S_* taxonomy names)
PLANE_ORDER = ["C0", "C1", "C2", "C3"]
PLANE_LABEL = {
    "C0": r"$S_{ctx}$",
    "C1": r"$S_{exec}$",
    "C2": r"$S_{cogn}$",
    "C3": r"$S_{arti}$",
}


# ---------------------------------------------------------------------------
# KV-cache sizing: bytes/token = 2 (K,V) * n_layers * n_kv_heads * head_dim * dtype_bytes
# GQA-aware (n_kv_heads, not n_heads). Used to convert token-native KV to bytes at replay.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelKV:
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype_bytes: float = 2.0  # fp16/bf16

    @property
    def kv_bytes_per_token(self) -> float:
        return 2.0 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes


MODELS = {
    # Llama-3-70B: 80 layers, GQA 8 KV heads, head_dim 128 -> 327,680 B/tok (320 KiB/tok)
    "llama3-70b": ModelKV("llama3-70b", n_layers=80, n_kv_heads=8, head_dim=128),
    # Llama-3-8B: 32 layers, GQA 8 KV heads, head_dim 128 -> 131,072 B/tok (128 KiB/tok)
    "llama3-8b": ModelKV("llama3-8b", n_layers=32, n_kv_heads=8, head_dim=128),
}
DEFAULT_MODEL = "llama3-70b"


# ---------------------------------------------------------------------------
# event
# ---------------------------------------------------------------------------
@dataclass
class Event:
    ts: float
    scenario: str
    plane: str
    obj_type: str
    op: str
    logical_bytes: int
    phys_bytes: int
    agent_id: str
    session_id: str
    req_id: str
    src: str


EVENT_FIELDS = [f.name for f in fields(Event)]


def write_events_csv(path: str, events: Iterable[Event]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        w.writeheader()
        for e in events:
            w.writerow(asdict(e))
            n += 1
    return n


# ---------------------------------------------------------------------------
# house plotting style (matches sim/characterize.py + plot_eval.py)
# ---------------------------------------------------------------------------
def house_style():
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        "font.size": 13, "font.weight": "bold",
        "axes.labelsize": 14, "axes.titlesize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    return sns.color_palette("colorblind")


# fixed, colorblind-safe color per plane (indices into sns 'colorblind')
def plane_colors():
    pal = house_style()
    # C0 blue, C1 orange, C2 green, C3 purple  (colorblind palette indices)
    return {"C0": pal[0], "C1": pal[1], "C2": pal[2], "C3": pal[4]}
