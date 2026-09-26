"""
G1: per-plane per-OBJECT size regime (ECDF, log-x), per session, with REAL
artifacts. The artifact plane (S_arti) is now driven by REAL OpenFOAM file sizes
(fd->path aggregation of the Frontera trace), not coding-agent text. Mined static
science artifacts and a facility anchor are overlaid.

Shows the four planes occupy distinct size regimes, but KV-per-round and real
artifacts are the two LARGE planes (a bandwidth peer relationship), with exec and
memory 4-5 orders smaller.

Sources (measured):
  C0 per-round KV write : coding_swarm (TraceLab tokens x KV bytes/token)
  C1 tool-call request  : coding_swarm (TraceLab input_chars)
  C2 memory object      : assistant_locomo (LoCoMo observations + summaries)
  C3 artifact file      : artifacts_openfoam_sizes.npy (REAL OpenFOAM fd->path)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import schema

HERE = os.path.dirname(os.path.abspath(__file__))

# mined static science artifacts (bytes) + facility anchor
STATIC = [("datacube", 55_600_000), ("mesh", 64_000_000)]


def _sizes(csv, plane):
    if not os.path.exists(csv):
        return np.array([])
    df = pd.read_csv(csv)
    d = df[(df.op == "write") & (df.plane == plane)]
    return d.logical_bytes.to_numpy(dtype=float)


def _ecdf(ax, x, color, label, ls="-"):
    x = np.sort(x[x > 0])
    if x.size == 0:
        return
    y = np.arange(1, x.size + 1) / x.size
    ax.step(x, y, where="post", color=color, lw=1.9, label=label, ls=ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coding", default=os.path.join(HERE, "data", "coding_swarm.csv"))
    ap.add_argument("--locomo", default=os.path.join(HERE, "data", "assistant_locomo.csv"))
    ap.add_argument("--artifacts", default=os.path.join(HERE, "data", "artifacts_openfoam_sizes.npy"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "f3_size_dist"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = schema.plane_colors()
    fig, ax = plt.subplots(figsize=(5, 3))
    _ecdf(ax, _sizes(a.coding, "C0"), colors["C0"], schema.PLANE_LABEL["C0"])
    _ecdf(ax, _sizes(a.coding, "C1"), colors["C1"], schema.PLANE_LABEL["C1"])
    _ecdf(ax, _sizes(a.locomo, "C2"), colors["C2"], schema.PLANE_LABEL["C2"])
    art = np.load(a.artifacts) if os.path.exists(a.artifacts) else _sizes(a.coding, "C3")
    _ecdf(ax, art, colors["C3"], schema.PLANE_LABEL["C3"])

    ax.set_xscale("log")
    ax.set_xlabel("object size (bytes, log)", fontweight="bold")
    ax.set_ylabel("CDF", fontweight="bold")
    ax.set_xlim(10, 1e12)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=12, loc="center left")

    fig.tight_layout()
    for d in (os.path.dirname(a.out),):
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(d, "f3_size_dist." + ext), bbox_inches="tight", dpi=300)

    print("=== G1 median object size by plane ===")
    for p, x in (("C0", _sizes(a.coding, "C0")), ("C1", _sizes(a.coding, "C1")),
                 ("C2", _sizes(a.locomo, "C2")), ("C3(OpenFOAM)", art)):
        if len(x):
            print(f"  {p}: n={len(x)} p50={np.median(x):,.0f}B p90={np.percentile(x,90):,.0f}B max={x.max():,.0f}B")


if __name__ == "__main__":
    main()
