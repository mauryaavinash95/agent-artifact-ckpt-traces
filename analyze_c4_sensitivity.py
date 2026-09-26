"""
C4-sensitivity: is cross-state co-occurrence an artifact of scenario composition?

Reviewers (AgenticAI4HPC'26, R1/R2/R3) asked whether the contention numbers reflect
how the traces were interleaved rather than real co-located correlation. The swarm
event stream keeps `agent_id`, so every composition knob can be swept post hoc from
the same CSV, with no re-extraction:

  1. co-location degree k -- subsample k sessions. k=1 is a SINGLE REAL SESSION with
     no composition at all, so it is the un-composed anchor for the claim.
  2. arrival window W      -- undo the original stagger (subtract each session's own
     start) and re-apply a fresh uniform(0, W) offset, i.e. exactly what
     extract_tracelab.py does at composition time. Swept over seeds.
  3. bin width             -- the analysis granularity itself.

Figure `f11b_cooccurrence_k` plots (1) with +-1 sd bands; (2) and (3) are printed and
reported in the text. The original bar chart (f11_cooccurrence) is left untouched.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import schema

HERE = os.path.dirname(os.path.abspath(__file__))

K_GRID = (1, 2, 4, 8, 16, 32, 48)
BIN_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
W_GRID = (0.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0)


def _shares(ts, planes, bin_s):
    """Share of ACTIVE bins holding >=2, >=3, and all 4 states."""
    b = (np.asarray(ts) // bin_s).astype(np.int64)
    n = pd.DataFrame({"b": b, "p": planes}).groupby("b")["p"].nunique()
    return (float((n >= 2).mean()) * 100,
            float((n >= 3).mean()) * 100,
            float((n == 4).mean()) * 100)


def sweep_k(d, bin_s=1.0, trials=20, seed=0):
    """k=1 uses every real session individually; k=48 is the full swarm."""
    rng = np.random.default_rng(seed)
    ags = np.array(sorted(d.agent_id.unique()))
    out = []
    for k in K_GRID:
        if k == 1:
            groups = [[a] for a in ags]
        elif k >= len(ags):
            groups = [ags]
        else:
            groups = [rng.choice(ags, k, replace=False) for _ in range(trials)]
        r = np.array([_shares(g.ts.values, g.plane.values, bin_s)
                      for g in (d[d.agent_id.isin(s)] for s in groups)])
        out.append((k, r.mean(0), r.std(0), len(groups)))
    return out


def sweep_bin(d):
    return [(b,) + _shares(d.ts.values, d.plane.values, b) for b in BIN_GRID]


def sweep_window(d, bin_s=1.0, seeds=5):
    """Undo the original stagger, re-apply uniform(0, W), as at composition time."""
    rel = d.ts - d.groupby("agent_id").ts.transform("min")
    ags = sorted(d.agent_id.unique())
    out = []
    for w in W_GRID:
        r = []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            off = {a: rng.uniform(0.0, w) for a in ags}
            r.append(_shares((rel + d.agent_id.map(off)).values, d.plane.values, bin_s))
        r = np.array(r)
        out.append((w, r.mean(0), r.std(0)))
    return out


def _fig(ks, out):
    import matplotlib.pyplot as plt
    pal = schema.house_style()
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.log2([k for k, _, _, _ in ks])
    series = [(0, ">=2 states", pal[0], "o"),
              (1, ">=3 states", pal[2], "s"),
              (2, "all 4 states", pal[4], "^")]
    for i, lab, c, m in series:
        mu = np.array([v[i] for _, v, _, _ in ks])
        sd = np.array([s[i] for _, _, s, _ in ks])
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
        ax.plot(x, mu, marker=m, color=c, lw=2, ms=6, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k, _, _, _ in ks])
    ax.set_xlabel("co-located agent sessions", fontweight="bold")
    ax.set_ylabel("active 1-s bins (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.annotate("one real session\n(no composition)", xy=(x[0], ks[0][1][0]),
                xytext=(x[0] + 0.15, 90), fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black"))
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for dd in (os.path.join(HERE, "figures"),):
        os.makedirs(dd, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(dd, f"{out}.{ext}"), bbox_inches="tight", dpi=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "data", "coding_swarm.csv"))
    ap.add_argument("--bin", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="f11b_cooccurrence_k")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)

    ks = sweep_k(d, a.bin, a.trials)
    _fig(ks, a.out)

    print("=== co-location degree (bin=%.2gs) ===" % a.bin)
    print(f"{'k':>4} {'n':>4} {'>=2':>14} {'>=3':>14} {'all4':>14}")
    for k, mu, sd, n in ks:
        print(f"{k:>4} {n:>4} " + " ".join(f"{m:>7.1f}+-{s:<5.1f}" for m, s in zip(mu, sd)))

    print("\n=== bin width (all sessions) ===")
    print(f"{'bin_s':>6} {'>=2':>8} {'>=3':>8} {'all4':>8}")
    for b, g2, g3, g4 in sweep_bin(d):
        print(f"{b:>6} {g2:>7.1f}% {g3:>7.1f}% {g4:>7.1f}%")

    print("\n=== arrival window (all sessions, 5 seeds) ===")
    print(f"{'W_s':>6} {'>=2':>14} {'>=3':>14} {'all4':>14}")
    for w, mu, sd in sweep_window(d, a.bin):
        print(f"{w:>6.0f} " + " ".join(f"{m:>7.1f}+-{s:<5.1f}" for m, s in zip(mu, sd)))


if __name__ == "__main__":
    main()
