"""
F6: write amplification of cognitive memory (C2) under the whole-file-rewrite
plus backup pattern that we measured in real agents.

Measured pattern (static block, Table II): A2MC memory/store.py and
open-ai-co-scientist run_store.py both persist the memory store by rewriting the
ENTIRE store file on every update, and A2MC additionally copies the prior file
to a .bak backup. We measured the memory-record size at 940 B (A2MC discoveries)
and the skill unit at a 7.7 KB median (SciLink).

Analytical consequence (this figure): if an agent accumulates records of size s
by rewriting the whole store on each of N updates, update k writes k*s bytes plus
a k*s backup, so the cumulative physical write is s*k*(k+1) ~ O(N^2). A delta or
append persists only s per update, so the ideal cumulative write is s*N ~ O(N).
The amplification factor is therefore (N+1), i.e. it grows without bound in the
number of memory updates. This is the evidence for delta and versioned memory.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import schema

HERE = os.path.dirname(os.path.abspath(__file__))


def cumulative(N, s, backup=True):
    k = np.arange(1, N + 1)
    mult = 2.0 if backup else 1.0
    phys = mult * s * np.cumsum(k)      # rewrite (+backup) of the whole store each update
    ideal = s * k                        # append/delta: only the new record
    return k, phys, ideal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "f6_write_amplification"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = schema.house_style()
    cg = schema.plane_colors()["C2"]

    # measured record sizes
    S_REC = 940.0     # A2MC memory record (bytes; 108,152 B / 115 discoveries)
    S_SKILL = 7675.0  # SciLink skill unit median (bytes; 41 skills at v0.0.60)

    k, phys_r, ideal_r = cumulative(a.updates, S_REC)
    _, phys_s, _ = cumulative(a.updates, S_SKILL)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(k, phys_r, color=cg, lw=2.0, label="rewrite+backup (940 B rec)")
    ax.plot(k, phys_s, color=cg, lw=2.0, ls="--", label="rewrite+backup (7.7 KB rec)")
    ax.plot(k, ideal_r, color="0.35", lw=1.6, ls=":", label="ideal delta/append")
    ax.fill_between(k, ideal_r, phys_r, color=cg, alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel("number of memory updates $N$", fontweight="bold")
    ax.set_ylabel("cumulative bytes written", fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=11, loc="upper left")

    Nend = a.updates
    amp = (phys_r[-1] / ideal_r[-1])
    ax.annotate(f"{amp:.0f}$\\times$ at $N$={Nend}",
                xy=(k[-1], phys_r[-1]), xytext=(k[-1] * 0.52, phys_r[-1] * 1.2),
                fontsize=11, fontweight="bold", color=cg)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight", dpi=300)

    print("=== F6 ===", a.out)
    print(f"  at N={Nend}: rewrite+backup(940B)={phys_r[-1]/1e6:.1f} MB, "
          f"ideal={ideal_r[-1]/1e3:.1f} KB, amplification={amp:.0f}x")


if __name__ == "__main__":
    main()
