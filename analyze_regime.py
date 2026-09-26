"""
G3: the two stress axes of the shared hierarchy -- object write-rate vs bytes per
object -- with each plane as a point. This replaces any single "which plane owns
the bytes" claim: the planes spread across a frequency x volume plane, and the
artifact plane has TWO operating regimes:

  sim-heavy  : a full-resolution run -> few, large writes (low-freq, high-volume)
  exploratory: coarse runs reported back to the LLM -> many, small writes
               (high-freq, low-volume)

So no single policy fits: one corner needs a bandwidth path, the other an
IOPS/metadata path. All quantities are measured (swarm rates + real OpenFOAM).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import schema

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coding", default=os.path.join(HERE, "data", "coding_swarm.csv"))
    ap.add_argument("--out", default="f5_regime")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(a.coding); w = d[d.op == "write"]
    dur = d.ts.max() - d.ts.min()
    col = schema.plane_colors()

    def pt(plane):
        s = w[w.plane == plane]
        return len(s) / dur, float(np.median(s.logical_bytes))

    pts = {p: pt(p) for p in ("C0", "C1", "C2", "C3")}
    of = json.load(open(os.path.join(HERE, "data", "artifacts_openfoam_stats.json")))
    sim_rate = of["artifact_rate"]; sim_sz = of["artifact_bytes"] / of["n_artifacts"]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    labels = {"C0": schema.PLANE_LABEL["C0"], "C1": schema.PLANE_LABEL["C1"],
              "C2": schema.PLANE_LABEL["C2"], "C3": schema.PLANE_LABEL["C3"] + " expl."}
    for p in ("C0", "C1", "C2", "C3"):
        r, b = pts[p]
        ax.scatter([r], [b], s=90, color=col[p], edgecolor="black", lw=0.7, zorder=3)
        ax.annotate(labels[p], (r, b), textcoords="offset points", xytext=(6, 5),
                    fontsize=11, fontweight="bold")
    # sim-heavy artifact regime (real OpenFOAM)
    ax.scatter([sim_rate], [sim_sz], s=120, color=col["C3"], edgecolor="black", lw=0.9,
               marker="D", zorder=3)
    ax.annotate(schema.PLANE_LABEL["C3"] + " sim", (sim_rate, sim_sz),
                textcoords="offset points", xytext=(6, 5), fontsize=11, fontweight="bold")
    # connect the two artifact regimes
    ax.plot([pts["C3"][0], sim_rate], [pts["C3"][1], sim_sz], color=col["C3"], ls="--", lw=1.0, alpha=0.7)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("object write-rate (writes/s, log)", fontweight="bold")
    ax.set_ylabel("bytes per object (log)", fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.text(0.03, 0.93, "low-freq,\nhigh-volume", transform=ax.transAxes, fontsize=10,
            color="0.35", va="top")
    ax.text(0.70, 0.30, "high-freq,\nlow-volume", transform=ax.transAxes, fontsize=10,
            color="0.35", ha="center")
    fig.tight_layout()
    for dd in (os.path.join(HERE, "figures"),):
        os.makedirs(dd, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(dd, f"{a.out}.{ext}"), bbox_inches="tight", dpi=300)
    print("=== G3 regime (write-rate, bytes/object) ===")
    for p in ("C0", "C1", "C2", "C3"):
        print(f"  {p}: rate={pts[p][0]:.2f}/s  size={pts[p][1]:,.0f}B")
    print(f"  C3 sim-heavy (OpenFOAM): rate={sim_rate:.3f}/s size={sim_sz:,.0f}B")


if __name__ == "__main__":
    main()
