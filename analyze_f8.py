"""
F8: C0 arrival burstiness in the reasoning scenario (real Mooncake serving).
KV writes arrive in sharp co-timed spikes rather than a smooth stream, so the
substrate must absorb bursts into a fast tier instead of serializing them to disk.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import schema

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "data", "reasoning_mooncake.csv"))
    ap.add_argument("--window", type=float, default=600.0)
    ap.add_argument("--bin-s", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "f8_c0_burstiness"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = schema.plane_colors()
    df = pd.read_csv(a.csv)
    d = df[df.op == "write"]
    t = d.ts.to_numpy(float)
    # global burstiness stats (all arrivals)
    ts = np.sort(t)
    iat = np.diff(ts)
    cv = iat.std(ddof=1) / iat.mean()
    cotimed = (iat == 0).mean()
    dur = ts[-1] - ts[0]
    nb = int(np.ceil(dur / a.bin_s))
    counts, _ = np.histogram(ts, bins=ts[0] + np.arange(nb + 1) * a.bin_s)
    peak, mean = counts.max(), counts[counts > 0].mean()

    # display window
    m = t <= a.window
    nbw = int(np.ceil(a.window / a.bin_s))
    edges = np.arange(nbw + 1) * a.bin_s
    centers = edges[:-1] + a.bin_s / 2
    gbytes, _ = np.histogram(t[m], bins=edges, weights=d.phys_bytes.to_numpy(float)[m])
    gbytes = gbytes / a.bin_s / 1e9

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.fill_between(centers, gbytes, color=colors["C0"], alpha=0.85, step="mid")
    ax.set_xlabel("scenario time (s)", fontweight="bold")
    ax.set_ylabel(r"$S_{ctx}$ KV write GB/s", fontweight="bold")
    ax.grid(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight", dpi=300)

    print(f"=== F8 === CV={cv:.3f} cotimed={cotimed:.3f} peak={peak} mean={mean:.2f} peak/mean={peak/mean:.2f}")


if __name__ == "__main__":
    main()
