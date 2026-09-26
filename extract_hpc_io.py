"""
Extract a REAL artifact-plane (S_arti) I/O stream from the HPC I/O traces
(Frontera fd-level syscall traces; OpenFOAM, GROMACS, ...).

Each per-process JSON is a list of syscall records:
  systemcall, type(datacall|metadatacall), timestamp(ISO us), pid, tid, node,
  descriptor(fd), path, new_path, size, return_value.
File descriptor -> path mapping:
  open/openat/creat/open64 : fd = return_value (int),  path in `path`/`new_path`
  fopen/fopen64/fdopen     : fd = descriptor   (int),  path in `path`
  dup/dup2                 : propagate path from src fd to dst fd
Bytes for read/write datacalls = return_value (actual bytes transferred).

We aggregate WRITE bytes PER PATH (fds are reused across files, so per-fd would
conflate files). Each artifact object = one written file (total bytes, closed at
its last write). We keep only real output paths (drop /proc,/sys,/dev,libs,...).

Outputs:
  out/artifacts_<app>.csv : one Event per artifact file (plane C3, op write)
  prints summary stats (file-size distribution, read/write, rate, burst).
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

import schema
from schema import Event

HERE = os.path.dirname(os.path.abspath(__file__))
HPC_ROOT = os.path.join(HERE, "data", "raw", "hpc_io")

OPEN_RET = {"openat", "open", "creat", "open64"}          # fd = return_value
OPEN_DESC = {"fopen", "fopen64", "fdopen", "freopen"}     # fd = descriptor
SYS_PREFIX = ("/proc", "/sys", "/dev", "/usr", "/lib", "/lib64", "/etc", "/opt",
              "/run", "/bin", "/sbin", "pipe:", "socket:", "anon_inode", "/tmp/ompi")


def _ts(s):
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _fd(r):
    sc = r["systemcall"]
    if sc in OPEN_RET:
        v = r.get("return_value")
        return v if isinstance(v, int) and v >= 0 else None
    if sc in OPEN_DESC:
        d = r.get("descriptor")
        return d if isinstance(d, int) and d >= 0 else None
    return None


def _iter(path):
    import ijson
    with open(path) as fh:
        for r in ijson.items(fh, "item"):
            yield r


def extract(app, case, run, top_k=12, scenario=None):
    run_dir = os.path.join(HPC_ROOT, app, case, str(run), "tracer")
    files = glob.glob(os.path.join(run_dir, "*.json"))
    files.sort(key=lambda f: os.path.getsize(f), reverse=True)
    files = files[:top_k]  # heaviest-writer processes

    per_path_bytes = defaultdict(int)     # path -> total write bytes (artifact size)
    per_path_last = {}                    # path -> last write timestamp
    per_path_nw = defaultdict(int)        # path -> write count
    read_bytes = write_bytes = 0
    n_read = n_write = 0
    tmin = tmax = None

    for fp in files:
        fd2path = {}
        for r in _iter(fp):
            sc = r.get("systemcall")
            ts = _ts(r.get("timestamp")) if r.get("timestamp") else None
            if ts is not None:
                tmin = ts if tmin is None or ts < tmin else tmin
                tmax = ts if tmax is None or ts > tmax else tmax
            if sc in OPEN_RET or sc in OPEN_DESC:
                fd = _fd(r)
                p = r.get("path") or r.get("new_path")
                if fd is not None and p:
                    fd2path[fd] = p
            elif sc in ("dup", "dup2"):
                src = r.get("descriptor"); dst = r.get("return_value")
                if isinstance(dst, int) and src in fd2path:
                    fd2path[dst] = fd2path[src]
            elif sc == "write":
                b = r.get("return_value") or 0
                if isinstance(b, int) and b > 0:
                    n_write += 1; write_bytes += b
                    p = fd2path.get(r.get("descriptor"))
                    if p and not p.startswith(SYS_PREFIX):
                        per_path_bytes[p] += b
                        per_path_nw[p] += 1
                        if ts is not None:
                            per_path_last[p] = ts
            elif sc == "read":
                b = r.get("return_value") or 0
                if isinstance(b, int) and b > 0:
                    n_read += 1; read_bytes += b

    # build artifact-object events (one per file)
    t0 = tmin or 0.0
    events = []
    sizes = []
    for p, b in per_path_bytes.items():
        sizes.append(b)
        ts_rel = round((per_path_last.get(p, t0) - t0), 3)
        otype = "artifact:" + (p.rsplit("/", 1)[-1].split(".")[-1] or "file")
        events.append(Event(ts=ts_rel, scenario=scenario or f"hpc_{app}", plane="C3",
                            obj_type=otype[:40], op="write", logical_bytes=int(b),
                            phys_bytes=int(b), agent_id=app, session_id=f"{app}/{case}/{run}",
                            req_id=os.path.basename(p), src=f"hpc:{app}"))
    events.sort(key=lambda e: e.ts)
    dur = (tmax - tmin) if (tmin and tmax) else 0.0
    sizes = np.array(sizes) if sizes else np.array([0])
    stats = dict(
        app=app, case=case, run=run, files_processed=len(files),
        n_artifacts=len(sizes), dur_s=dur,
        size_p50=float(np.median(sizes)), size_p90=float(np.percentile(sizes, 90)),
        size_p99=float(np.percentile(sizes, 99)), size_max=float(sizes.max()),
        total_write_bytes=int(write_bytes), total_read_bytes=int(read_bytes),
        n_write=n_write, n_read=n_read,
        rw_ratio=(read_bytes / write_bytes) if write_bytes else float("nan"),
        artifact_bytes=int(sizes.sum()),
        write_rate=(n_write / dur) if dur else float("nan"),
        artifact_rate=(len(sizes) / dur) if dur else float("nan"),
    )
    return events, sizes, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="openfoam")
    ap.add_argument("--case", default="incompressible")
    ap.add_argument("--run", default="1")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    events, sizes, st = extract(a.app, a.case, a.run, a.top_k)
    out = a.out or os.path.join(HERE, "data", f"artifacts_{a.app}.csv")
    schema.write_events_csv(out, events)
    import json
    np.save(os.path.join(HERE, "data", f"artifacts_{a.app}_sizes.npy"), sizes)
    with open(os.path.join(HERE, "data", f"artifacts_{a.app}_stats.json"), "w") as fh:
        json.dump(st, fh, indent=1)
    print(f"=== HPC artifact I/O: {a.app}/{a.case}/{a.run} (top {st['files_processed']} writers) ===")
    print(f"  wrote {len(events)} artifact-file events -> {out}")
    print(f"  run duration = {st['dur_s']:.0f}s   artifacts = {st['n_artifacts']}")
    print(f"  artifact FILE size: p50={st['size_p50']:,.0f}B p90={st['size_p90']:,.0f}B "
          f"p99={st['size_p99']:,.0f}B max={st['size_max']:,.0f}B")
    print(f"  total artifact bytes = {st['artifact_bytes']/1e6:.1f}MB  "
          f"write_syscalls={st['n_write']:,} read_syscalls={st['n_read']:,}")
    print(f"  read:write byte ratio = {st['rw_ratio']:.1f}:1   "
          f"artifact-object rate = {st['artifact_rate']:.3f}/s   write-syscall rate = {st['write_rate']:.1f}/s")


if __name__ == "__main__":
    main()
