#!/usr/bin/env bash
# Fetch the third-party traces this artifact is derived from.
#
# The normalized event streams in data/*.csv are already committed, so the nine
# paper figures regenerate with at most the Mooncake trace (8.6 MB):
#
#   ./fetch_data.sh --figures-only   # Mooncake only  -> make figures
#   ./fetch_data.sh --all            # everything     -> make extract
#
# Nothing here is redistributed by us; each dataset is pulled from upstream.
# Licenses and attribution: docs/PROVENANCE.md
set -euo pipefail

RAW="$(cd "$(dirname "$0")" && pwd)/data/raw"
MODE="${1:---figures-only}"

have() { command -v "$1" >/dev/null 2>&1; }
have git || { echo "error: git is required" >&2; exit 1; }

# Shallow-clone one subdirectory of a repo (no history, no unrelated files).
sparse_clone() {   # <url> <ref> <subdir> <dest>
  local url=$1 ref=$2 sub=$3 dest=$4
  [ -d "$dest" ] && { echo "  have $(basename "$dest"), skipping"; return; }
  local tmp; tmp=$(mktemp -d)
  git clone --depth 1 --branch "$ref" --filter=blob:none --sparse "$url" "$tmp/r" >/dev/null 2>&1
  git -C "$tmp/r" sparse-checkout set "$sub" >/dev/null 2>&1
  mkdir -p "$(dirname "$dest")"
  cp -R "$tmp/r/$sub" "$dest"
  rm -rf "$tmp"
}

echo "==> Mooncake serving trace (Apache-2.0) -> data/raw/mooncake"
sparse_clone https://github.com/kvcache-ai/Mooncake.git main \
             FAST25-release/traces "$RAW/mooncake"

if [ "$MODE" = "--figures-only" ]; then
  echo
  echo "Done. All nine paper figures can now be regenerated:  make figures"
  exit 0
fi

echo "==> LoCoMo conversational memory -> data/raw/locomo"
sparse_clone https://github.com/snap-research/locomo.git main data "$RAW/locomo_tmp"
mkdir -p "$RAW/locomo" && [ -f "$RAW/locomo_tmp/locomo10.json" ] \
  && mv "$RAW/locomo_tmp/locomo10.json" "$RAW/locomo/" && rm -rf "$RAW/locomo_tmp" || true

echo "==> PerLTQA long-term memory -> data/raw/perltqa"
sparse_clone https://github.com/Elvin-Yiming-Du/PerLTQA.git main \
             Dataset/en "$RAW/perltqa_tmp"
mkdir -p "$RAW/perltqa" && [ -f "$RAW/perltqa_tmp/perltmem_en.json" ] \
  && mv "$RAW/perltqa_tmp/perltmem_en.json" "$RAW/perltqa/" && rm -rf "$RAW/perltqa_tmp" || true

echo "==> TraceLab coding-agent traces (CC BY 4.0, ~100 MB) -> data/raw/tracelab"
mkdir -p "$RAW/tracelab"
if [ ! -f "$RAW/tracelab/syfi_coding_trace.duckdb" ]; then
  echo "    The DuckDB build is published as a GitHub Release asset."
  echo "    Download syfi_coding_trace.duckdb from"
  echo "      https://github.com/uw-syfi/TraceLab/releases"
  echo "    into $RAW/tracelab/"
else
  echo "  have tracelab duckdb, skipping"
fi

cat <<EOF

==> Remaining sources are large or access-gated; see docs/PROVENANCE.md:
    - Nebius SWE-agent-trajectories (~1 GB, HuggingFace) -> data/raw/hf/
      only needed to rebuild data/artifact_pool_sizes.npy (already committed)
    - HPC I/O traces, Zenodo 10.5281/zenodo.17428437     -> data/raw/hpc_io/
      only needed to rebuild data/artifacts_openfoam_* (already committed)

Done.  make extract   # re-derive data/*.csv
       make figures   # regenerate the nine paper figures
EOF
