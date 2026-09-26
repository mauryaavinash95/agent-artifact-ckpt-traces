"""
Extract a normalized C0 (KV) event stream from the real Mooncake serving trace
(KV-centric disaggregated serving). Each request line is
{timestamp(ms), input_length(tok), output_length(tok), hash_ids[512-tok blocks]}.

We emit one C0 KV write per request, sized (input+output) tokens x KV bytes/token
at the request's real arrival time. This is the "reasoning / long-context" clock:
KV is the only durable-scale state, arrivals are bursty and often co-timed, and
prefix blocks (hash_ids) are reused across requests.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import schema
from schema import Event

TRACES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "mooncake")


def extract(trace="toolagent_trace.jsonl", model=schema.DEFAULT_MODEL,
            limit=0, scenario="reasoning"):
    kv = schema.MODELS[model].kv_bytes_per_token
    path = os.path.join(TRACES, trace)
    events, ts_list = [], []
    seen_blocks = set()
    reuse_hits = reuse_tot = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            r = json.loads(line)
            t = float(r.get("timestamp", 0)) / 1000.0  # ms -> s
            itok = int(r.get("input_length", 0))
            otok = int(r.get("output_length", 0))
            wtok = itok + otok
            if wtok <= 0:
                continue
            b = int(wtok * kv)
            events.append(Event(ts=round(t, 3), scenario=scenario, plane="C0",
                                obj_type="kv", op="write", logical_bytes=b, phys_bytes=b,
                                agent_id="serve", session_id=str(r.get("hash_ids", [None])[0]),
                                req_id=str(i), src="mooncake"))
            ts_list.append(t)
            # prefix reuse accounting from hash blocks
            for h in r.get("hash_ids", []):
                reuse_tot += 1
                if h in seen_blocks:
                    reuse_hits += 1
                else:
                    seen_blocks.add(h)
    events.sort(key=lambda e: e.ts)
    stats = _arrival_stats(np.asarray(ts_list, dtype=float))
    stats["prefix_reuse_frac"] = reuse_hits / reuse_tot if reuse_tot else float("nan")
    return events, stats


def _arrival_stats(t):
    t = np.sort(t[np.isfinite(t)])
    if t.size < 2:
        return {}
    iat = np.diff(t)
    dur = float(t[-1] - t[0]) or 1.0
    nb = max(1, int(np.ceil(dur)))
    counts, _ = np.histogram(t, bins=t[0] + np.arange(nb + 1))
    nz = counts[counts > 0]
    return {
        "n": int(t.size),
        "duration_s": dur,
        "mean_rate_rps": float(t.size / dur),
        "iat_cv": float(iat.std(ddof=1) / iat.mean()) if iat.mean() else float("nan"),
        "frac_cotimed": float((iat == 0).mean()),
        "peak_per_s": int(counts.max()),
        "peak_over_mean": float(counts.max() / nz.mean()) if nz.size else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="toolagent_trace.jsonl")
    ap.add_argument("--model", default=schema.DEFAULT_MODEL, choices=list(schema.MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reasoning_mooncake.csv"))
    a = ap.parse_args()
    events, stats = extract(a.trace, a.model, a.limit)
    n = schema.write_events_csv(a.out, events)
    print(f"wrote {n} C0 events -> {a.out}")
    print("  arrival stats:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()})


if __name__ == "__main__":
    main()
