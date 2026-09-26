PY ?= python

.PHONY: help figures verify extract static clean

help:
	@echo "make figures   regenerate the nine paper figures into figures/"
	@echo "               needs: ./fetch_data.sh --figures-only   (8.6 MB)"
	@echo "make verify    reprint the paper's headline numbers from data/"
	@echo "make static    check the static agent measurements (offline)"
	@echo "               add ARGS=--clone to re-measure from pinned upstreams"
	@echo "make extract   re-derive data/*.csv from raw traces"
	@echo "               needs: ./fetch_data.sh --all"
	@echo "make clean     remove __pycache__"

# --- the nine figures in the paper -----------------------------------------
# Fig 2  f3_size_dist            Fig 3  f12_artifact_io
# Fig 4  f8_c0_burstiness        Fig 5  f10a_kv_lru
# Fig 6  f10b_kv_skew            Fig 7  f5_regime
# Fig 8  f11b_cooccurrence_k     Fig 9  f2_embed_dup
# Fig 10 f6_write_amplification
figures:
	$(PY) analyze_f3.py
	$(PY) analyze_artio.py
	$(PY) analyze_f8.py
	$(PY) analyze_c6_kvreuse.py
	$(PY) analyze_regime.py
	$(PY) analyze_c4_sensitivity.py
	$(PY) analyze_f2.py
	$(PY) analyze_f6.py
	@echo
	@echo "figures/ updated."

verify:
	@echo "=== size regimes (paper Sec. V-A) ==="
	@$(PY) analyze_regime.py 2>/dev/null | tail -6
	@echo "=== KV reuse and eviction (Sec. V-C) ==="
	@$(PY) analyze_c6_kvreuse.py 2>/dev/null | tail -5
	@echo "=== cross-state co-occurrence (Sec. V-E) ==="
	@$(PY) analyze_c4_sensitivity.py 2>/dev/null | tail -8
	@echo "=== write amplification (Sec. V-F) ==="
	@$(PY) analyze_f6.py 2>/dev/null | tail -3

static:
	$(PY) measure_agents.py $(ARGS)

extract:
	$(PY) extract_tracelab.py
	$(PY) extract_mooncake.py
	$(PY) extract_locomo.py
	$(PY) compose_sci_session.py

clean:
	rm -rf __pycache__ docs/reference/__pycache__
