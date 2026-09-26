"""
G2: the artifact state (S_arti) size regime, pooled across REAL general-purpose
and scientific traces (tool results, screenshots, DOM, code patches, eval logs,
simulation fields, mined science artifacts), equal-weight per type. Single
measured CDF, no fit. I/O character (read-heavy, bursty) is reported in the text.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import schema

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="f12_artifact_io")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    import matplotlib.colors as mc

    base = schema.plane_colors()["C3"]
    dark = tuple(0.6 * c for c in mc.to_rgb(base))  # darker shade of the S_arti color

    pool = np.load(os.path.join(HERE, "data", "artifact_pool_sizes.npy"))
    x = np.sort(pool[pool > 0].astype(float))
    y = np.arange(1, x.size + 1) / x.size

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.step(x, y, where="post", color=dark, lw=2.6)

    ax.set_xscale("log")
    ax.set_xlim(100, 64 * 2**20)
    ax.set_ylim(0, 1.0)
    ticks = [1024, 4096, 65536, 1048576, 8 * 1048576, 64 * 1048576]
    labs = ["1K", "4K", "64K", "1M", "8M", "64M"]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labs))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("artifact size", fontweight="bold")
    ax.set_ylabel("CDF", fontweight="bold")
    ax.grid(False)

    fig.tight_layout()
    for d in (os.path.join(HERE, "figures"),):
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(d, f"{a.out}.{ext}"), bbox_inches="tight", dpi=300)
    print(f"=== G2 pooled artifact CDF: n={x.size} p50={np.median(x):,.0f}B "
          f"p90={np.percentile(x,90):,.0f}B p99={np.percentile(x,99):,.0f}B max={x.max():,.0f}B ===")


if __name__ == "__main__":
    main()
