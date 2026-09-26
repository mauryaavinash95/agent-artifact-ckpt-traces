"""
Compose a single scientific-agent SESSION (not the 48-agent swarm) in two regimes,
by attaching REAL OpenFOAM artifact I/O to a REAL TraceLab agent-loop backbone.

Backbone (measured): one real TraceLab coding session provides the S_ctx (per-round
KV), S_exec (tool-call), and S_cogn (memory) events with real intra-session timing.
Artifacts (measured): real OpenFOAM per-file writes (artifacts_openfoam.csv) provide
the S_arti stream, mapped onto the session window.

Two regimes on the KV:artifact axis (composition labeled, per the grounding ladder):
  sim-heavy   : few agent turns launch ONE full-resolution run -> full artifact stream.
  exploratory : many agent turns, each a COARSE run -> only small artifact files.

Emits out/sci_session_simheavy.csv and out/sci_session_explore.csv.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import schema
from schema import Event

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")


def _session_backbone(coding_csv, agent, n_rounds):
    d = pd.read_csv(coding_csv)
    d = d[d.agent_id == agent].copy()
    d["ts"] = d.ts - d.ts.min()
    rounds = list(dict.fromkeys(d[d.plane == "C0"].sort_values("ts").req_id))[:n_rounds]
    d = d[d.req_id.isin(rounds) | (d.plane == "C2")]
    win = d.ts.max() or 1.0
    return d, win


def _artifacts(coarse_only=False, coarse_kb=100):
    p = os.path.join(OUT, "artifacts_openfoam.csv")
    a = pd.read_csv(p)
    a = a[a.op == "write"]
    if coarse_only:
        a = a[a.logical_bytes <= coarse_kb * 1024]
    return a


def compose(agent, n_rounds, coarse, scenario, seed=7):
    back, win = _session_backbone(os.path.join(OUT, "coding_swarm.csv"), agent, n_rounds)
    arts = _artifacts(coarse_only=coarse)
    rng = np.random.default_rng(seed)
    evs = []
    for _, r in back.iterrows():
        evs.append(Event(ts=round(float(r.ts), 3), scenario=scenario, plane=r.plane,
                         obj_type=r.obj_type, op=r.op, logical_bytes=int(r.logical_bytes),
                         phys_bytes=int(r.phys_bytes), agent_id="sci", session_id=scenario,
                         req_id=str(r.req_id), src=r.src))
    # place artifact writes across the session window (real sizes, spread over the run)
    n = len(arts)
    if n:
        # map the artifact stream's own relative time onto the session window
        at = arts.ts.to_numpy(dtype=float)
        at = (at - at.min()) / (at.max() - at.min() + 1e-9) * win
        for (ts_new, (_, r)) in zip(at, arts.iterrows()):
            evs.append(Event(ts=round(float(ts_new), 3), scenario=scenario, plane="C3",
                             obj_type=r.obj_type, op="write", logical_bytes=int(r.logical_bytes),
                             phys_bytes=int(r.phys_bytes), agent_id="sci", session_id=scenario,
                             req_id=str(r.req_id), src="hpc:openfoam"))
    evs.sort(key=lambda e: e.ts)
    out = os.path.join(OUT, f"{scenario}.csv")
    schema.write_events_csv(out, evs)
    # summary
    df = pd.DataFrame([e.__dict__ for e in evs])
    w = df[df.op == "write"]
    by_b = w.groupby("plane").logical_bytes.sum()
    by_o = w.groupby("plane").size()
    print(f"=== {scenario}: agent={agent} rounds={n_rounds} coarse={coarse} events={len(evs)} win={win:.0f}s ===")
    print("  bytes/plane:", {k: f"{v/1e6:.2f}MB" for k, v in by_b.items()})
    print("  ops/plane  :", dict(by_o))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="a028")
    a = ap.parse_args()
    # sim-heavy: few turns + full-resolution artifact stream
    compose(a.agent, n_rounds=8, coarse=False, scenario="sci_session_simheavy")
    # exploratory: many turns + coarse (small-file) artifacts only
    compose(a.agent, n_rounds=40, coarse=True, scenario="sci_session_explore")


if __name__ == "__main__":
    main()
