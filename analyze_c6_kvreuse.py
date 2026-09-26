"""
C5c + C6: KV prefix-reuse opportunity and lifetime-vs-value misalignment.

Reads the REAL Mooncake tool-agent serving trace (per-request 512-token block
hash_ids) and measures, at block granularity:

  (C5c) reuse opportunity   : fraction of block references that are re-references
                              (KV that a shared substrate could reuse instead of
                              recomputing).
  (C6)  lifetime misalignment:
        - LRU miss/hit-ratio curve: hit-rate over ALL references as a function of
          resident cache capacity (Mattson stack distance). Recency at a realistic
          KV capacity captures only a fraction of the reuse, because reuse
          distances are heavy-tailed.
        - reuse skew (Lorenz): a tiny hot set of blocks holds most references, so
          a value/schedule-aware policy captures the reuse a recency-only cache
          at the same capacity drops.

Two-panel single-column figure: (a) LRU hit-rate vs capacity, (b) reuse skew.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import schema

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(HERE, "data", "raw", "mooncake")


def _load_refs(trace):
    refs = []
    with open(os.path.join(TRACES, trace)) as f:
        for line in f:
            refs.extend(json.loads(line).get("hash_ids", []))
    return refs


class _BIT:
    def __init__(self, n):
        self.n = n
        self.t = [0] * (n + 1)

    def upd(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & -i

    def q(self, i):
        i += 1
        r = 0
        while i > 0:
            r += self.t[i]
            i -= i & -i
        return r


def _stack_distances(refs):
    """Mattson LRU stack distance per reference (-1 = compulsory miss)."""
    n = len(refs)
    bit = _BIT(n + 1)
    last = {}
    sd = np.empty(n, dtype=np.int64)
    t = 0
    for k, b in enumerate(refs):
        if b in last:
            s = last[b]
            sd[k] = bit.q(t - 1) - bit.q(s)
            bit.upd(s, -1)
        else:
            sd[k] = -1
        bit.upd(t, 1)
        last[b] = t
        t += 1
    return sd


def analyze(trace="toolagent_trace.jsonl", model=schema.DEFAULT_MODEL):
    refs = _load_refs(trace)
    n = len(refs)
    arr = np.asarray(refs)
    # reuse skew
    _, counts = np.unique(arr, return_counts=True)
    counts = np.sort(counts)[::-1]
    uniq = counts.size
    reuse_frac = 1.0 - uniq / n
    # LRU stack distances -> hit-rate vs capacity
    sd = _stack_distances(refs)
    finite = sd[sd >= 0]
    block_gib = schema.MODELS[model].kv_bytes_per_token * 512 / 2 ** 30
    caps = np.unique(np.round(np.logspace(0, np.log10(uniq), 40)).astype(int))
    hit = np.array([(finite < C).sum() / n for C in caps])
    # skew Lorenz: cumulative share of refs vs fraction of (hot-sorted) blocks
    cum = np.cumsum(counts) / n
    kfrac = np.arange(1, uniq + 1) / uniq
    stats = dict(n=n, uniq=uniq, reuse_frac=reuse_frac, block_gib=block_gib,
                 sd_p50=float(np.median(finite)), sd_p90=float(np.percentile(finite, 90)),
                 sd_p99=float(np.percentile(finite, 99)))
    return dict(caps=caps, hit=hit, kfrac=kfrac, cum=cum, counts=counts,
                block_gib=block_gib, reuse_frac=reuse_frac, stats=stats)


def _save(fig, out):
    for d in (os.path.join(HERE, "figures"),):
        if os.path.isdir(os.path.dirname(d)) or os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(d, f"{out}.{ext}"), bbox_inches="tight", dpi=300)


def _fig(a, out):
    """Two independent figures: f10a (LRU hit vs capacity) and f10b (reuse skew).
    No panel titles, no in-figure callouts; the numbers are explained in the text."""
    import matplotlib.pyplot as plt
    schema.house_style()
    cz = schema.plane_colors()["C0"]

    # --- f10a: LRU hit-rate vs resident KV capacity, with the reuse ceiling ---
    gib = a["caps"] * a["block_gib"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogx(gib, a["hit"] * 100, color=cz, lw=2.4)
    ax.axhline(a["reuse_frac"] * 100, ls="--", color="0.4", lw=1.4)
    ax.set_xlabel("resident KV capacity (GiB, log)", fontweight="bold")
    ax.set_ylabel("KV hit rate (%)", fontweight="bold")
    ax.set_ylim(0, 62)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "f10a_kv_lru")

    # --- f10b: reuse skew (cumulative refs vs hottest-first block fraction) ---
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.semilogx(a["kfrac"] * 100, a["cum"] * 100, color=cz, lw=2.4)
    ax2.set_xlabel("fraction of blocks, hottest first (%)", fontweight="bold")
    ax2.set_ylabel("cumulative share of refs (%)", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(False)
    fig2.tight_layout()
    _save(fig2, "f10b_kv_skew")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="toolagent_trace.jsonl")
    ap.add_argument("--model", default=schema.DEFAULT_MODEL, choices=list(schema.MODELS))
    ap.add_argument("--out", default="f10_kv_reuse")
    args = ap.parse_args()
    a = analyze(args.trace, args.model)
    _fig(a, args.out)
    s = a["stats"]
    print("=== C5c/C6 KV reuse ===")
    print(f"  refs={s['n']:,} unique={s['uniq']:,} reuse_frac={s['reuse_frac']*100:.1f}%")
    print(f"  stack-dist p50={s['sd_p50']:,.0f} p90={s['sd_p90']:,.0f} p99={s['sd_p99']:,.0f}")
    print(f"  block={s['block_gib']*1024:.0f} MiB ({args.model})")
    gib = a["caps"] * a["block_gib"]
    for target in (10, 40, 160, 640):
        print(f"  hit@{target:>4} GiB = {np.interp(target, gib, a['hit'])*100:.1f}%")


if __name__ == "__main__":
    main()
