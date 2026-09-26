"""
F2: embedding duplication. The same corpus is embedded once per co-resident
model variant, so at-rest vector-database capacity scales with the number of
variants. Measured from the mined repos (summary.txt):

  Foam-Agent (one ~65 MB OpenFOAM tutorial corpus) ships THREE co-resident
  embedding variants, each with its own FAISS index and its own copy of the
  chunk docstore:
    Qwen3-0.6B (1024-d): 3.6 MB index + 31.1 MB docstore
    text-embedding-3-small (1536-d): 5.4 MB index + 129.6 MB docstore
    Qwen3-8B (4096-d): 14.4 MB index + 129.6 MB docstore
  A2MC ships TWO co-resident ChromaDB model-version profiles: 19 MB and 43 MB.

A deduplicated, content-shared layer would keep one copy of the content plus the
indices actually needed, collapsing the multiplier to about one.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import schema

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, index_MB, docstore_MB) measured on disk
FOAM = [("0.6B/1024-d", 3.6, 31.1), ("3-small/1536-d", 5.4, 129.6), ("8B/4096-d", 14.4, 129.6)]
A2MC = [("api-31", 8.0, 11.0), ("api-43", 14.0, 29.0)]  # chroma split ~ (hnsw, sqlite); totals 19,43 MB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "f2_embed_dup"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cp = schema.plane_colors()["C3"]
    fig, ax = plt.subplots(figsize=(5, 3))

    groups = [("Foam-Agent\n(1 corpus, 3 variants)", FOAM),
              ("A2MC\n(2 profiles)", A2MC)]
    x = np.arange(len(groups))
    w = 0.55
    shades = [0.45, 0.7, 1.0]
    for gi, (gname, variants) in enumerate(groups):
        bottom = 0.0
        for vi, (lab, idx, doc) in enumerate(variants):
            tot = idx + doc
            ax.bar(gi, tot, w, bottom=bottom, color=cp, alpha=shades[vi % len(shades)],
                   edgecolor="black", lw=0.6)
            ax.text(gi, bottom + tot / 2, lab, ha="center", va="center", fontsize=10,
                    color="black")
            bottom += tot
        # single-variant dedup target = largest single variant
        single = max(idx + doc for _l, idx, doc in variants)
        ax.hlines(single, gi - w / 2, gi + w / 2, color="black", ls="--", lw=1.2)
        ax.annotate(f"{bottom/single:.1f}$\\times$", xy=(gi, bottom),
                    xytext=(gi, bottom * 1.04), ha="center", fontsize=11,
                    fontweight="bold", color=cp)

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=12)
    ax.set_ylabel("at-rest vector-DB size (MB)", fontweight="bold")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="black", ls="--", label="single-variant (dedup target)")],
              fontsize=11, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight", dpi=300)

    foam_tot = sum(i + d for _l, i, d in FOAM)
    a2mc_tot = sum(i + d for _l, i, d in A2MC)
    print(f"=== F2 === Foam 3 variants total={foam_tot:.1f}MB  A2MC 2 profiles total={a2mc_tot:.1f}MB")


if __name__ == "__main__":
    main()
