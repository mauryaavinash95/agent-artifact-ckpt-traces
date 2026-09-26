# Provenance

Where every number in the paper comes from, and how much of it is measured
rather than modelled.

## Grounding levels

The paper (Table III) labels each quantity. We repeat the rule here because it
is the main thing a reader should check:

| Level | Meaning |
|---|---|
| **measured** | read directly out of a trace or a committed file |
| **derived** | computed from a measured quantity plus one stated constant (e.g. tokens x KV bytes/token) |
| **composed** | placed on a shared clock by us, because no public trace records all four states together |

No composed quantity is reported as measured. The composition parameters are
listed in the paper (Sec. IV) and swept in Sec. V-E.

## The one composition caveat that matters

No public trace records all four states on one clock, so the co-located swarm
is assembled by us. `extract_tracelab.py` selects its sessions with

```sql
HAVING rounds BETWEEN 15 AND 300 ORDER BY tcalls DESC LIMIT 48
```

i.e. the **48 heaviest sessions by tool-call count**, median 328 tool calls
against a population median of 25 across all 4,265 TraceLab sessions. The
co-occurrence rates are therefore a **provisioning bound** for the sessions a
shared substrate must absorb, not a typical-case rate. `analyze_c4_sensitivity.py`
sweeps the degree of co-location down to k=1 (a single real session, no
composition at all) precisely so this can be separated.

## Third-party sources

None of these are redistributed here. `fetch_data.sh` pulls them from upstream.

| Source | Upstream | License | Used for |
|---|---|---|---|
| TraceLab coding-agent traces | https://github.com/uw-syfi/TraceLab | CC BY 4.0 (data), Apache-2.0 (code) | `data/coding_swarm.csv`; execution and artifact timing |
| Mooncake serving trace | https://github.com/kvcache-ai/Mooncake | Apache-2.0 | `data/reasoning_mooncake.csv`; KV arrivals and 512-token prefix reuse |
| LoCoMo | https://github.com/snap-research/locomo | see upstream | `data/assistant_locomo.csv`; cognitive-memory object sizes |
| PerLTQA | https://github.com/Elvin-Yiming-Du/PerLTQA | see upstream | memory consolidation ratio |
| Nebius SWE-agent-trajectories | HuggingFace `nebius/SWE-agent-trajectories` | see upstream | artifact size pool |
| Mind2Web | `mm2web_sample.jsonl` | see upstream | artifact size pool |
| HPC I/O traces (Frontera) | Zenodo [10.5281/zenodo.17428437](https://doi.org/10.5281/zenodo.17428437) | see upstream | real OpenFOAM/GROMACS artifact file sizes |

Please cite the originals if you use the derived streams.

## Figure -> script -> input

| Paper figure | Script | Input |
|---|---|---|
| Fig. 2 `f3_size_dist` | `analyze_f3.py` | `data/coding_swarm.csv`, `data/assistant_locomo.csv`, `data/artifacts_openfoam_sizes.npy` |
| Fig. 3 `f5_regime` | `analyze_regime.py` | `data/coding_swarm.csv` |
| Fig. 4 `f8_c0_burstiness` | `analyze_f8.py` | `data/reasoning_mooncake.csv` |
| Fig. 5 `f12_artifact_io` | `analyze_artio.py` | `data/artifact_pool_sizes.npy` |
| Fig. 6 `f10a_kv_lru` | `analyze_c6_kvreuse.py` | `data/raw/mooncake/toolagent_trace.jsonl` |
| Fig. 7 `f10b_kv_skew` | `analyze_c6_kvreuse.py` | same |
| Fig. 8 `f11b_cooccurrence_k` | `analyze_c4_sensitivity.py` | `data/coding_swarm.csv` |
| Fig. 9 `f2_embed_dup` | `analyze_f2.py` | measured constants (see `measure_agents.py`) |
| Fig. 10 `f6_write_amplification` | `analyze_f6.py` | measured constants (940 B record, 7.6 kB skill) |

Fig. 1 is a hand-drawn schematic and has no script.

`analyze_f9.py` produces no paper figure but computes the consolidation ratios
quoted in Sec. V-F (4.1x on LoCoMo, 39.7x on PerLTQA).

## Derived, not measured

* **KV bytes.** Traces record tokens, not bytes. We convert with one stated
  model, Llama-3-70B: `2 x 80 layers x 8 KV heads x 128 head_dim x 2 B =
  327,680 B/token = 320 KiB/token` (`schema.py:MODELS`). Retarget by changing
  `DEFAULT_MODEL`; a 512-token block is 160 MiB at this setting.
* **Write amplification** (Fig. 10) is the analytical consequence of a measured
  pattern: A2MC and open-ai-co-scientist rewrite the whole store on every
  update and A2MC also copies a `.bak`, so update *k* writes *k* records twice.
  The record size (940 B) is measured; the `O(N^2)` curve is arithmetic.

## Resolution limits worth knowing

* **Mooncake timestamps are quantised to about 3.05 s** (1,180 distinct stamps
  over 23,608 requests; smallest gap between distinct stamps 3,049 ms). So
  "95% of requests are co-timed" is partly an artefact of that bucket width,
  and burstiness below ~3 s is not observable in this trace. We therefore treat
  it as a lower bound on burstiness.
* **The OpenFOAM artifact stream** is a file-descriptor-to-path reconstruction
  over the heaviest-writing ranks, not every file the run touched.

## Static agent measurements

`measure_agents.py` records the footprint of the four scientific agents and can
re-derive all of it from pinned upstream revisions:

```
python measure_agents.py            # print the recorded table (offline)
python measure_agents.py --clone    # re-measure and diff
```

Pins are in `docs/static_measurements.json`. Two details are easy to get wrong:

* **Foam-Agent's 314 MB vector store is Git-LFS.** A normal clone reports ~131 B
  per file (the pointer). The tool sets `GIT_LFS_SKIP_SMUDGE=1` and reads the
  `size` field out of each pointer, so the real 313,635,093 B is measured from a
  ~43 MB checkout instead of a ~900 MB one.
* **The corpus `.txt` is LFS-tracked too**, so the 248-case count is recovered
  from FAISS geometry: an `IndexFlat` file is a 45-byte header plus `n*d`
  float32s, and the three shipped indices embed the same corpus at 1024, 1536
  and 4096 dimensions. All three yield exactly n = 248, which is a mutual
  cross-check rather than an assumption.

SciLink is pinned to the **PyPI sdist 0.0.60** (sha256 `37b6664431...`) because
0.0.60 has no git tag. Its `examples/` are not shipped in any sdist, so the
33 kB spectrum and 55.6 MB datacube are measured from the repository at commit
`59279854`.
