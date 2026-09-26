"""
Pool REAL artifact-object sizes across general-purpose and scientific agent
traces, one artifact type at a time, equal-weight per type (sample up to K each)
so the resulting distribution reflects the artifact-state size *regime* rather
than the frequency of any one corpus.

Types pooled (all measured):
  tool result   TraceLab tool-call results (coding_swarm C3)
  screenshot    Multimodal-Mind2Web screenshot_png_bytes
  DOM/HTML      Multimodal-Mind2Web cleaned_html_bytes
  code patch    Nebius SWE-agent generated_patch length
  eval log      Nebius SWE-agent eval_logs length
  sim field     OpenFOAM per-file writes (fd->path)
  sci artifact  mined datacube/mesh/spectrum/image sizes
Writes out/artifact_pool_sizes.npy and prints per-type + pooled stats.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
HF = os.path.join(HERE, "..", "..", "data", "raw", "hf")
K = 400
RNG = np.random.default_rng(7)


def _samp(x, k=K):
    x = np.asarray([v for v in x if v and v > 0], dtype=float)
    if x.size > k:
        x = RNG.choice(x, k, replace=False)
    return x


def tracelab():
    d = pd.read_csv(os.path.join(OUT, "coding_swarm.csv"))
    return _samp(d[(d.plane == "C3") & (d.op == "write")].logical_bytes.tolist())


def mind2web():
    scr, dom = [], []
    p = os.path.join(HF, "mm2web_sample.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            scr.append(r.get("screenshot_png_bytes"))
            dom.append(r.get("cleaned_html_bytes"))
    return _samp(scr), _samp(dom)


def nebius():
    import pyarrow.parquet as pq
    patch, elog = [], []
    fs = sorted(glob.glob(os.path.join(HF, "SWE-agent-trajectories", "data", "*.parquet")))
    if fs:
        t = pq.read_table(fs[0], columns=["generated_patch", "eval_logs"]).to_pydict()
        for v in t.get("generated_patch", []):
            if v:
                patch.append(len(v.encode("utf-8", "ignore")))
        for v in t.get("eval_logs", []):
            if v:
                elog.append(len(v.encode("utf-8", "ignore")))
    return _samp(patch), _samp(elog)


def openfoam():
    p = os.path.join(OUT, "artifacts_openfoam_sizes.npy")
    return _samp(np.load(p)) if os.path.exists(p) else np.array([])


def mined():
    return np.array([32_900, 262_000, 55_600_000, 64_000_000], dtype=float)


def main():
    types = {}
    types["tool result"] = tracelab()
    scr, dom = mind2web()
    types["screenshot"] = scr
    types["DOM/HTML"] = dom
    patch, elog = nebius()
    types["code patch"] = patch
    types["eval log"] = elog
    types["sim field"] = openfoam()
    types["sci artifact"] = mined()

    pool = np.concatenate([v for v in types.values() if v.size])
    np.save(os.path.join(OUT, "artifact_pool_sizes.npy"), pool)

    def fmt(v):
        for u, s in ((2**30, "G"), (2**20, "M"), (2**10, "K")):
            if v >= u:
                return f"{v/u:.1f}{s}"
        return f"{v:.0f}B"
    print("=== pooled artifact sizes ===")
    for t, v in types.items():
        if v.size:
            print(f"  {t:12s} n={v.size:4d} p50={fmt(np.median(v)):>7} "
                  f"p90={fmt(np.percentile(v,90)):>7} max={fmt(v.max()):>7}")
    print(f"  POOL n={pool.size}  p50={fmt(np.median(pool))} p90={fmt(np.percentile(pool,90))} "
          f"p99={fmt(np.percentile(pool,99))} min={fmt(pool.min())} max={fmt(pool.max())}")


if __name__ == "__main__":
    main()
