"""
Extract a normalized C2 (cognitive memory) event stream from the real LoCoMo
long-conversation benchmark (10 multi-session conversations), and measure the
consolidation/compaction that defines the memory lifecycle.

Per conversation we walk the sessions in order. Each session contributes:
  - raw dialogue turns (the input that memory is distilled from),
  - per-speaker observations (the incremental memory writes),
  - one session summary (the periodic consolidation write).
We emit an observation write per observation and a summary write per session, and
we report the compaction ratio (raw dialogue bytes / consolidated bytes).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import schema
from schema import Event

LOCOMO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "locomo", "locomo10.json")
INTER_TURN_S = 2.0      # nominal spacing to place turns on a clock
INTER_SESSION_S = 60.0  # nominal gap between sessions


def _obs_items(node):
    """Flatten a session observation node {speaker:[items]} into text strings."""
    out = []
    if isinstance(node, dict):
        for _spk, items in node.items():
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, (list, tuple)) and it:
                        out.append(str(it[0]))
                    else:
                        out.append(str(it))
    return out


def extract(scenario="assistant"):
    data = json.load(open(LOCOMO))
    events = []
    comp_rows = []  # per-conversation (dialogue_bytes, consolidated_bytes, turns, obs)
    for ci, s in enumerate(data):
        conv = s["conversation"]
        obs = s.get("observation", {})
        summ = s.get("session_summary", {})
        sessions = [k for k in conv if k.startswith("session_") and not k.endswith("date_time")]
        # order sessions numerically
        def _idx(k):
            try:
                return int(k.split("_")[1])
            except Exception:
                return 0
        sessions.sort(key=_idx)

        t = 0.0
        agent = f"c{ci:02d}"
        dia_bytes = cons_bytes = n_turns = n_obs = 0
        for k in sessions:
            turns = conv[k] if isinstance(conv[k], list) else []
            si = _idx(k)
            obs_texts = _obs_items(obs.get(f"session_{si}_observation", {}))
            summ_text = str(summ.get(f"session_{si}_summary", ""))
            # distribute observations across this session's turns
            n = max(1, len(turns))
            obs_per_turn = len(obs_texts) / n
            acc = 0.0
            oi = 0
            for turn in turns:
                dia_bytes += len(str(turn.get("text", "")))
                n_turns += 1
                acc += obs_per_turn
                while acc >= 1.0 and oi < len(obs_texts):
                    b = max(1, len(obs_texts[oi]))
                    events.append(Event(ts=round(t, 3), scenario=scenario, plane="C2",
                                        obj_type="mem_obs", op="write",
                                        logical_bytes=b, phys_bytes=b,
                                        agent_id=agent, session_id=k, req_id=f"o{oi}",
                                        src="locomo"))
                    cons_bytes += b
                    n_obs += 1
                    oi += 1
                    acc -= 1.0
                t += INTER_TURN_S
            # session summary consolidation write
            if summ_text:
                b = len(summ_text)
                events.append(Event(ts=round(t, 3), scenario=scenario, plane="C2",
                                    obj_type="mem_summary", op="write",
                                    logical_bytes=b, phys_bytes=b,
                                    agent_id=agent, session_id=k, req_id="summary",
                                    src="locomo"))
                cons_bytes += b
            t += INTER_SESSION_S
        comp_rows.append((dia_bytes, cons_bytes, n_turns, n_obs, len(sessions)))

    events.sort(key=lambda e: e.ts)
    return events, comp_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "assistant_locomo.csv"))
    a = ap.parse_args()
    events, comp = extract()
    n = schema.write_events_csv(a.out, events)
    dia = np.array([c[0] for c in comp], float)
    cons = np.array([c[1] for c in comp], float)
    turns = np.array([c[2] for c in comp], float)
    nobs = np.array([c[3] for c in comp], float)
    ratio = dia / np.maximum(cons, 1)
    print(f"wrote {n} C2 events -> {a.out}")
    print(f"  conversations={len(comp)}  turns/conv mean={turns.mean():.0f}  "
          f"obs/conv mean={nobs.mean():.0f}  obs/turn={nobs.sum()/turns.sum():.2f}")
    print(f"  dialogue->consolidated compaction: mean={ratio.mean():.1f}x "
          f"min={ratio.min():.1f}x max={ratio.max():.1f}x")


if __name__ == "__main__":
    main()
