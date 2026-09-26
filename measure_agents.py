"""
Static footprint of the four open-source scientific agents (paper Sec. III-A,
Table II, Table IV).

These numbers do not come from a trace; they are measured from the agents' own
committed code and data. This script makes that measurement reproducible.

    python measure_agents.py            # offline: print the recorded table
    python measure_agents.py --clone    # re-measure from the pinned revisions

--clone fetches the three git sources with GIT_LFS_SKIP_SMUDGE=1 and the SciLink
PyPI sdist, recomputes every value, and diffs against docs/static_measurements.json.

Why skip LFS smudging: Foam-Agent ships its 314 MB vector store through Git LFS.
A normal checkout downloads ~900 MB of payload; a pointer-only checkout is ~43 MB
and each pointer still carries the real byte count, e.g.

    version https://git-lfs.github.com/spec/v1
    oid sha256:f563345f...
    size 4063277

so summing pointer sizes measures the store exactly. Reading the checked-out file
size instead yields ~131 B per file, which is the classic way to get this wrong.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# published embedding widths of the three models Foam-Agent ships indices for
FOAM_EMBED_DIMS = {
    "Qwen_Qwen3-Embedding-0.6B": 1024,
    "text-embedding-3-small": 1536,
    "Qwen_Qwen3-Embedding-8B": 4096,
}
RECORD = os.path.join(HERE, "docs", "static_measurements.json")

SDIST_URL = (
    "https://files.pythonhosted.org/packages/b6/22/"
    "45ec46c2ab5423b68e72c63f34adb208963cec3d1d988f9e556442ed7caa/scilink-0.0.60.tar.gz"
)


# --------------------------------------------------------------------------- io
def load_record():
    with open(RECORD) as f:
        return json.load(f)


def lfs_size(path):
    """Real byte count of a file: the LFS pointer's `size` if it is a pointer."""
    try:
        with open(path, "rb") as f:
            head = f.read(200)
    except OSError:
        return 0
    if head.startswith(b"version https://git-lfs"):
        m = re.search(rb"^size (\d+)$", head, re.M)
        if m:
            return int(m.group(1))
    return os.path.getsize(path)


def tree_bytes(root):
    return sum(
        lfs_size(os.path.join(d, n))
        for d, _sub, files in os.walk(root)
        for n in files
    )


def clone(url, ref, dest, sparse=None):
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    cmd = ["git", "clone", "--depth", "1", "--branch", ref]
    if sparse:
        cmd += ["--filter=blob:none", "--sparse"]
    subprocess.run(cmd + [url, dest], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if sparse:
        subprocess.run(["git", "-C", dest, "sparse-checkout", "set", sparse],
                       check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sha = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.strip()
    return sha


def fetch_sdist(dest):
    tgz = os.path.join(dest, "scilink.tar.gz")
    urllib.request.urlretrieve(SDIST_URL, tgz)
    digest = hashlib.sha256(open(tgz, "rb").read()).hexdigest()
    with tarfile.open(tgz) as t:
        t.extractall(dest)
    return digest, os.path.join(dest, "scilink-0.0.60")


# ------------------------------------------------------------------- measurers
def measure_foam(r):
    out = {}
    # tag v2.0.0 lays the corpus out as database/{raw,faiss}; later main nests it
    # under database/foundation-v10/. Accept whichever is present.
    struct, faiss_dir = None, None
    for base in ("database", "database/foundation-v10"):
        s = os.path.join(r, base, "raw", "openfoam_tutorials_structure.txt")
        f = os.path.join(r, base, "faiss")
        if os.path.exists(s) and os.path.isdir(f):
            struct, faiss_dir = s, f
            break
    if struct:
        txt = open(struct, errors="ignore").read()
        if "<case_begin>" in txt:                 # only when LFS content is present
            out["tutorial_cases"] = txt.count("<case_begin>")
    faiss_dir = faiss_dir or ""
    if os.path.isdir(faiss_dir):
        out["vector_store_bytes"] = tree_bytes(faiss_dir)
        # The corpus .txt is itself LFS-tracked, so under a pointer-only checkout we
        # recover the document count from FAISS geometry instead. An IndexFlat file is
        # a 45-byte header followed by n*d float32s, and the three shipped indices
        # embed the same corpus at three published dimensionalities, so each must
        # yield the same integer n -- a mutual cross-check, not an assumption.
        counts = []
        for model, dim in sorted(FOAM_EMBED_DIMS.items(), key=lambda kv: kv[1]):
            idx = os.path.join(faiss_dir, model, "openfoam_tutorials_details", "index.faiss")
            if os.path.exists(idx):
                n = (lfs_size(idx) - 45) / (4 * dim)
                if n == int(n):
                    counts.append((int(n), dim))
        if counts and len({n for n, _d in counts}) == 1:
            out.setdefault("tutorial_cases", counts[0][0])
            out["embedding_dims"] = sorted(d for _n, d in counts)
    cfg = os.path.join(r, "src/config.py")
    if os.path.exists(cfg):
        m = re.search(r"max_loop\s*:\s*int\s*=\s*(\d+)", open(cfg).read())
        if m:
            out["max_loop"] = int(m.group(1))
    mesh = os.path.join(r, "tandem_wing.msh")
    if os.path.exists(mesh):
        out["input_mesh_bytes"] = lfs_size(mesh)
    utils = os.path.join(r, "src/utils.py")
    if os.path.exists(utils):
        out["vector_backend"] = "faiss" if "FAISS" in open(utils).read() else "unknown"
    return out


def measure_a2mc(r):
    out = {}
    nodes, edges = [], []
    for prof in ("api-31-0", "api-43-1"):
        g = os.path.join(r, "rag/graphs", prof + ".json")
        if os.path.exists(g):
            d = json.load(open(g))
            nodes.append(len(d.get("nodes", [])))
            edges.append(len(d.get("links", d.get("edges", []))))
    if nodes:
        out["graph_nodes_min"], out["graph_nodes_max"] = min(nodes), max(nodes)
        out["graph_edges_min"], out["graph_edges_max"] = min(edges), max(edges)

    dirs, bins, chunks = [], [], []
    for prof in ("api-31-0", "api-43-1"):
        cd = os.path.join(r, "rag/chroma_db", prof)
        if os.path.isdir(cd):
            dirs.append(tree_bytes(cd))
            for d, _s, files in os.walk(cd):
                if "data_level0.bin" in files:
                    bins.append(lfs_size(os.path.join(d, "data_level0.bin")))
        meta = os.path.join(r, "rag/metadata", prof + ".json")
        if os.path.exists(meta):
            c = json.load(open(meta)).get("stats", {}).get("chunk_count")
            if c:
                chunks.append(c)
    if dirs:
        out["chroma_dir_bytes_min"], out["chroma_dir_bytes_max"] = min(dirs), max(dirs)
    if bins:
        out["data_level0_bytes_min"], out["data_level0_bytes_max"] = min(bins), max(bins)
    if chunks:
        out["chunks_min"], out["chunks_max"] = min(chunks), max(chunks)

    vs = os.path.join(r, "rag/vector_store.py")
    if os.path.exists(vs) and "all-MiniLM-L6-v2" in open(vs).read():
        out["embedding_dim"] = 384

    disc = os.path.join(r, "use_cases/ELM-FATES_Kougarok/memory/gained_knowledge/discoveries.json")
    if os.path.exists(disc):
        d = json.load(open(disc))
        recs = [v for v in d.values() if isinstance(v, dict)]
        if recs:
            out["discovery_mean_bytes"] = round(os.path.getsize(disc) / len(recs))
    mgr = os.path.join(r, "memory/manager.py")
    if os.path.exists(mgr):
        src = open(mgr).read()
        out["typed_json_stores"] = sum(
            src.count(f'"{n}.json"') > 0
            for n in ("discoveries", "experiments", "parameters", "failed_approaches")
        )
    out["exec_log_file"] = "workflow_log.json" if _grep(r, "workflow_log.json") else "unknown"
    return out


def measure_cosci(r):
    out = {}
    models = os.path.join(r, "app/models.py")
    if os.path.exists(models):
        src = open(models).read()
        m = re.search(r"class Hypothesis.*?def __init__\(self.*?\n(.*?)(?=\n    def |\nclass )", src, re.S)
        if m:
            out["hypothesis_fields"] = len(re.findall(r"^\s+self\.(\w+)", m.group(1), re.M))
    agents = os.path.join(r, "app/agents.py")
    if os.path.exists(agents):
        cls = re.findall(r"^class (\w+)", open(agents).read(), re.M)
        out["worker_agents"] = len([c for c in cls if c.endswith("Agent") and c != "SupervisorAgent"])
    logs = sorted(
        lfs_size(os.path.join(r, "results", n))
        for n in (os.listdir(os.path.join(r, "results")) if os.path.isdir(os.path.join(r, "results")) else [])
        if n.startswith("log_") and n.endswith(".txt")
    )
    if logs:
        out["run_log_bytes_min"], out["run_log_bytes_max"] = logs[0], logs[-1]
    return out


def measure_scilink(r):
    out = {}
    sk = os.path.join(r, "scilink", "skills")
    if not os.path.isdir(sk):
        return out
    sizes, pys = [], 0
    for d, _s, files in os.walk(sk):
        for n in files:
            p = os.path.join(d, n)
            if n.endswith(".py"):
                pys += 1
            elif n.endswith(".md") and os.path.basename(d) == n[:-3]:
                sizes.append(os.path.getsize(p))
    if sizes:
        sizes.sort()
        out["skill_count"] = len(sizes)
        out["skill_bytes_min"] = sizes[0]
        out["skill_bytes_max"] = sizes[-1]
        out["skill_bytes_median"] = round(statistics.median(sizes))
    out["companion_py_files"] = pys
    kb = os.path.join(r, "scilink/knowledge/knowledge_base.py")
    if os.path.exists(kb):
        out["retrieval_backend"] = "faiss" if "import faiss" in open(kb).read() else "unknown"
    loader = os.path.join(sk, "loader.py")
    uses_db = os.path.exists(loader) and "sqlite3" in open(loader, errors="ignore").read()
    out["memory_store_index"] = "sqlite" if uses_db else "none"
    return out


def measure_scilink_git(r):
    """examples/ ship only in the repo, never in a PyPI sdist."""
    out = {}
    for key, rel in (("spectrum_bytes", "examples/eels_identification_demo/spectrum.npy"),
                     ("datacube_bytes", "examples/eels_plasmons_demo/datacube.npy")):
        p = os.path.join(r, rel)
        if os.path.exists(p):
            out[key] = lfs_size(p)
    return out


def _grep(root, needle):
    for d, _s, files in os.walk(root):
        for n in files:
            if n.endswith((".py", ".md", ".json")):
                try:
                    if needle in open(os.path.join(d, n), errors="ignore").read():
                        return True
                except OSError:
                    pass
    return False


MEASURERS = {
    "foam-agent": measure_foam,
    "a2mc": measure_a2mc,
    "open-ai-co-scientist": measure_cosci,
    "scilink": measure_scilink,
    "scilink-git": measure_scilink_git,
}


# ------------------------------------------------------------------------ main
def show(rec):
    print(f"{'agent':<22} {'quantity':<24} {'recorded':>14}  paper")
    print("-" * 96)
    for m in rec["measurements"]:
        print(f"{m['agent']:<22} {m['key']:<24} {str(m['value']):>14}  {m['paper']}")
    print("\npins:")
    for a, p in rec["pins"].items():
        print(f"  {a:<22} {p['ref']:<10} {p.get('commit', p.get('sha256', ''))[:40]}")
    print("\nRe-measure from the pinned revisions with:  python measure_agents.py --clone")


def verify(rec, workdir):
    os.makedirs(workdir, exist_ok=True)
    fresh = {}
    for agent, pin in rec["pins"].items():
        dest = os.path.join(workdir, agent)
        print(f"==> {agent} @ {pin['ref']}")
        try:
            if agent == "scilink":
                if not os.path.isdir(os.path.join(dest, "scilink-0.0.60")):
                    os.makedirs(dest, exist_ok=True)
                    digest, _ = fetch_sdist(dest)
                    if digest != pin["sha256"]:
                        print(f"    !! sdist sha256 mismatch\n       got {digest}")
                    else:
                        print("    sdist sha256 ok")
                root = os.path.join(dest, "scilink-0.0.60")
            else:
                if not os.path.isdir(dest):
                    sha = clone(pin["url"], pin["ref"], dest,
                                sparse="examples" if agent == "scilink-git" else None)
                    if sha != pin["commit"]:
                        print(f"    !! commit drift: pinned {pin['commit'][:12]}, got {sha[:12]}")
                root = dest
            fresh[agent] = MEASURERS[agent](root)
        except Exception as e:                                   # noqa: BLE001
            print(f"    skipped ({type(e).__name__}: {e})")
            fresh[agent] = {}

    print(f"\n{'agent':<22} {'quantity':<24} {'recorded':>14} {'measured':>14}  status")
    print("-" * 92)
    ok = miss = bad = 0
    for m in rec["measurements"]:
        got = fresh.get(m["agent"], {}).get(m["key"], None)
        if got is None:
            status, miss = "-- not measured", miss + 1
        elif got == m["value"]:
            status, ok = "MATCH", ok + 1
        else:
            status, bad = "MISMATCH", bad + 1
        print(f"{m['agent']:<22} {m['key']:<24} {str(m['value']):>14} {str(got):>14}  {status}")
    print(f"\n{ok} match, {bad} mismatch, {miss} not measured")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clone", action="store_true",
                    help="re-measure from the pinned upstream revisions")
    ap.add_argument("--workdir", default=os.path.join(HERE, "data", "raw", "agents"),
                    help="where to place the clones (kept, so re-runs are offline)")
    ap.add_argument("--clean", action="store_true", help="remove the clones and exit")
    a = ap.parse_args()

    if a.clean:
        shutil.rmtree(a.workdir, ignore_errors=True)
        print("removed", a.workdir)
        return 0
    rec = load_record()
    if not a.clone:
        show(rec)
        return 0
    return verify(rec, a.workdir)


if __name__ == "__main__":
    sys.exit(main())
