# The Agentic Memory Hierarchy — artifact

Extractors, normalized event streams, and analysis scripts for

> **The Agentic Memory Hierarchy: Cross-State I/O Characterization for a Unified Substrate**
> Avinash Maurya, Bogdan Nicolae, M. Mustafa Rafique
> AgenticAI4HPC, SC26 Workshops, Chicago, IL, 2026

An LLM agent produces four kinds of state on one memory-storage hierarchy: the
model context (KV cache), execution state, cognitive memory, and bulk artifacts.
Each sits behind a separate substrate today. This repository holds the
measurement pipeline behind that characterization.

## Quickstart

```bash
pip install -r requirements.txt
./fetch_data.sh --figures-only     # 8.6 MB: the Mooncake trace
make figures                       # regenerate all nine paper figures
make verify                        # reprint the paper's headline numbers
```

The normalized event streams in `data/` are committed (9 MB), so nothing else
needs downloading to reproduce the figures.

## What is here

```
schema.py                 one event schema + the reference KV model
extract_*.py              raw trace  -> normalized events
compose_sci_session.py    scenario composition
analyze_*.py              normalized events -> figures and numbers
measure_agents.py         static footprint of the four scientific agents
data/                     normalized event streams (committed)
figures/                  the nine paper figures
docs/PROVENANCE.md        what is measured, derived, or composed
```

Every extractor emits the same `Event` record, so a new trace becomes a new
source by writing one extractor, and a new scenario by writing one composition.
Replay is deterministic given a seed. The reference model is one parameter:

```python
# schema.py
MODELS = {"llama3-70b": ..., "llama3-8b": ...}   # 2*L*H*d*bytes per token
DEFAULT_MODEL = "llama3-70b"                     # 327,680 B/token = 320 KiB
```

## Figures

| Paper | File | Script | What it shows |
|---|---|---|---|
| Fig. 2 | `f3_size_dist` | `analyze_f3.py` | the four states occupy distinct size regimes |
| Fig. 3 | `f12_artifact_io` | `analyze_artio.py` | artifact sizes are heavy-tailed |
| Fig. 4 | `f8_c0_burstiness` | `analyze_f8.py` | KV arrives in co-timed spikes |
| Fig. 5 | `f10a_kv_lru` | `analyze_c6_kvreuse.py` | 55% of KV blocks are reusable, LRU captures 34% |
| Fig. 6 | `f10b_kv_skew` | `analyze_c6_kvreuse.py` | reuse is concentrated in a few hot blocks |
| Fig. 7 | `f5_regime` | `analyze_regime.py` | two stress axes: bytes/object vs object rate |
| Fig. 8 | `f11b_cooccurrence_k` | `analyze_c4_sensitivity.py` | states collide even in a single un-composed session |
| Fig. 9 | `f2_embed_dup` | `analyze_f2.py` | embeddings duplicated per model variant |
| Fig. 10 | `f6_write_amplification` | `analyze_f6.py` | whole-file-rewrite memory grows O(N^2) |

Fig. 1 is a hand-drawn schematic.

## Reproducing the static agent numbers

Section III-A and Table IV are measured from four real agents rather than from a
trace. That measurement is reproducible:

```bash
python measure_agents.py            # print the recorded table
python measure_agents.py --clone    # re-measure from pinned upstreams and diff
```

`--clone` fetches Foam-Agent `v2.0.0`, A2MC and open-ai-co-scientist at pinned
commits, and the SciLink 0.0.60 sdist, then checks all 33 recorded values. It
reads Git-LFS pointer sizes rather than downloading payloads, so Foam-Agent's
314 MB vector store is verified from a ~43 MB checkout.

## Re-deriving the event streams

```bash
./fetch_data.sh --all
make extract
```

`data/*.csv` are regenerated from the raw traces. Two inputs are large or
access-gated (Nebius trajectories, the Zenodo HPC I/O traces); the outputs they
feed are already committed. See `docs/PROVENANCE.md`.

## Important notes

Read `docs/PROVENANCE.md` before quoting anything. Three key points:

1. **The 48-session swarm is composed by us**, because no public trace records
   all four states on one clock. The sessions are the **heaviest** in TraceLab
   (median 328 tool calls vs 25 across the corpus), so the co-occurrence rates
   are a provisioning bound, not a typical rate. `analyze_c4_sensitivity.py`
   sweeps down to a single un-composed session for exactly this reason.
2. **KV bytes are derived**, not measured: traces record tokens, and we convert
   with one stated model.
3. **The Mooncake trace is time-quantised to about 3.05 s**, so it bounds
   burstiness from below; sub-bucket structure is not observable in it.

## License

Code is MIT (`LICENSE`). The third-party traces are **not** redistributed here;
`fetch_data.sh` pulls them from upstream and they keep their own licenses
(TraceLab CC BY 4.0, Mooncake Apache-2.0, others listed in
`docs/PROVENANCE.md`). Please cite the original datasets.

## Citation

```bibtex
@inproceedings{maurya2026agentic,
  title     = {The Agentic Memory Hierarchy: Cross-State {I/O} Characterization
               for a Unified Substrate},
  author    = {Maurya, Avinash and Nicolae, Bogdan and Rafique, M. Mustafa},
  booktitle = {SC26 Workshops: Agentic AI for HPC (AgenticAI4HPC)},
  year      = {2026}
}
```
