"""
Extract a normalized cross-plane (C0-C3) event stream from the REAL TraceLab
coding-agent trace (Claude Code + Codex, syfi_coding_trace.duckdb).

Grounding ladder for this scenario ("coding_swarm"):
  MEASURED (per-object sizes + intra-session timing):
    C0 KV     : tokens/round (newly_append + output) x KV bytes/token(model)   [rounds]
    C1 exec   : tool-call request bytes = input_chars, at real emitted_at       [tool_calls]
    C3 artifact: tool-call result bytes = result_chars, at real result_at        [tool_calls]
  DERIVED (trace-grounded cadence, sizes from LoCoMo/PerLTQA):
    C2 memory : ~0.43 observation writes/round + 1 summary/session
  COMPOSED+LABELED (scenario clock):
    N real sessions overlaid; per-session idle gaps capped (IDLE_CAP), and each
    session given a staggered arrival over ARRIVAL_WINDOW -> a co-located swarm.

Only reads the trace; converts token-native KV to bytes at replay via schema.MODELS.
"""
from __future__ import annotations

import argparse
import os
import random

import duckdb

import schema
from schema import Event

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "tracelab", "syfi_coding_trace.duckdb")

# C2 sizes measured from LoCoMo (data/locomo/locomo10.json)
LOCOMO_OBS_BYTES = 88       # observation avg chars
LOCOMO_SUMMARY_BYTES = 653  # session summary avg chars
C2_OBS_PER_ROUND = 0.43     # LoCoMo observations per turn


def _select_sessions(con, k, min_rounds, max_rounds):
    rows = con.execute(
        """
        SELECT r.session_id,
               count(DISTINCT r.round_pk) AS rounds,
               count(tc.tool_index)       AS tcalls
        FROM rounds r LEFT JOIN tool_calls tc USING(round_pk)
        GROUP BY r.session_id
        HAVING rounds BETWEEN ? AND ?
        ORDER BY tcalls DESC
        LIMIT ?
        """,
        [min_rounds, max_rounds, k],
    ).fetchall()
    return [r[0] for r in rows]


def _placeholders(n):
    return ",".join(["?"] * n)


def _fetch_session_data(con, session_ids):
    ph = _placeholders(len(session_ids))
    rounds = con.execute(
        f"""
        SELECT round_pk, session_id, provider, round_index,
               COALESCE(prefix_tokens,0), COALESCE(newly_append_tokens,0),
               COALESCE(output_tokens,0)
        FROM rounds WHERE session_id IN ({ph})
        """,
        session_ids,
    ).fetchall()

    rstart = dict(
        con.execute(
            f"""
            SELECT te.round_pk, min(te.timestamp)
            FROM timing_events te JOIN rounds r USING(round_pk)
            WHERE r.session_id IN ({ph})
            GROUP BY te.round_pk
            """,
            session_ids,
        ).fetchall()
    )

    tcs = con.execute(
        f"""
        SELECT tc.round_pk, tc.tool_index, tc.tool_name,
               tc.emitted_at, tc.result_at,
               COALESCE(tc.input_chars,0), COALESCE(tc.result_chars,0)
        FROM tool_calls tc JOIN rounds r USING(round_pk)
        WHERE r.session_id IN ({ph})
        ORDER BY tc.round_pk, tc.tool_index
        """,
        session_ids,
    ).fetchall()

    tcs_by_round = {}
    for row in tcs:
        tcs_by_round.setdefault(row[0], []).append(row[1:])
    return rounds, rstart, tcs_by_round


def _sec(dt_a, dt_b):
    """seconds between two datetimes (a - b); None-safe."""
    if dt_a is None or dt_b is None:
        return None
    return (dt_a - dt_b).total_seconds()


def extract(k=48, min_rounds=15, max_rounds=300, model=schema.DEFAULT_MODEL,
            idle_cap=60.0, arrival_window=120.0, seed=7, scenario="coding_swarm"):
    kv_per_tok = schema.MODELS[model].kv_bytes_per_token
    con = duckdb.connect(DB, read_only=True)
    session_ids = _select_sessions(con, k, min_rounds, max_rounds)
    rounds, rstart, tcs_by_round = _fetch_session_data(con, session_ids)
    con.close()

    # group rounds by session
    by_sess = {}
    for (rpk, sid, prov, ridx, ptok, ntok, otok) in rounds:
        by_sess.setdefault(sid, []).append((rpk, prov, ridx, ptok, ntok, otok))

    rng = random.Random(seed)
    events = []
    for ai, sid in enumerate(session_ids):
        rs = sorted(by_sess.get(sid, []), key=lambda x: x[2])  # by round_index
        if not rs:
            continue
        agent = f"a{ai:03d}"

        # --- build absolute-timed raw events for this session ---
        raw = []  # (abs_dt, plane, obj_type, op, logical, phys, req_id)
        obs_acc = 0.0
        last_round_dt = None
        for (rpk, prov, ridx, ptok, ntok, otok) in rs:
            r_dt = rstart.get(rpk)
            tclist = tcs_by_round.get(rpk, [])
            if r_dt is None and tclist:  # fallback: first tool emitted_at
                r_dt = min((t[2] for t in tclist if t[2] is not None), default=None)
            if r_dt is not None:
                last_round_dt = r_dt
                # C0: KV written for newly computed tokens (prefill new + decode)
                wtok = int(ntok) + int(otok)
                if wtok > 0:
                    b = int(wtok * kv_per_tok)
                    raw.append((r_dt, "C0", "kv", "write", b, b, str(rpk)))
                # C0: prefix reuse = cache read (carried; not used by F4 write panel)
                if int(ptok) > 0:
                    b = int(int(ptok) * kv_per_tok)
                    raw.append((r_dt, "C0", "kv", "read", b, b, str(rpk)))
                # C2: derived memory observation writes at LoCoMo cadence
                obs_acc += C2_OBS_PER_ROUND
                while obs_acc >= 1.0:
                    raw.append((r_dt, "C2", "mem_obs", "write",
                                LOCOMO_OBS_BYTES, LOCOMO_OBS_BYTES, str(rpk)))
                    obs_acc -= 1.0
            # C1 request + C3 result per tool call
            for (tidx, tname, emit, res, ichars, rchars) in tclist:
                t1 = emit or res or r_dt
                t3 = res or emit or r_dt
                if t1 is not None and int(ichars) >= 0:
                    raw.append((t1, "C1", f"tool:{tname}", "write",
                                int(ichars), int(ichars), str(rpk)))
                if t3 is not None and int(rchars) > 0:
                    raw.append((t3, "C3", f"artifact:{tname}", "write",
                                int(rchars), int(rchars), str(rpk)))
        # C2: one session summary write at end
        if last_round_dt is not None:
            raw.append((last_round_dt, "C2", "mem_summary", "write",
                        LOCOMO_SUMMARY_BYTES, LOCOMO_SUMMARY_BYTES, "session"))

        if not raw:
            continue
        raw.sort(key=lambda x: x[0])
        t0 = raw[0][0]

        # --- relative seconds + idle-gap compression ---
        rel = [_sec(r[0], t0) for r in raw]
        comp = [0.0] * len(rel)
        for i in range(1, len(rel)):
            gap = rel[i] - rel[i - 1]
            if gap < 0:
                gap = 0.0
            comp[i] = comp[i - 1] + min(gap, idle_cap)

        # --- staggered arrival for this session (composed swarm) ---
        arrival = rng.uniform(0.0, arrival_window)

        for i, r in enumerate(raw):
            events.append(Event(
                ts=round(arrival + comp[i], 3),
                scenario=scenario,
                plane=r[1], obj_type=r[2], op=r[3],
                logical_bytes=r[4], phys_bytes=r[5],
                agent_id=agent, session_id=sid, req_id=r[6],
                src="tracelab",
            ))

    events.sort(key=lambda e: e.ts)
    return events, session_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=48)
    ap.add_argument("--min-rounds", type=int, default=15)
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--model", default=schema.DEFAULT_MODEL, choices=list(schema.MODELS))
    ap.add_argument("--idle-cap", type=float, default=60.0)
    ap.add_argument("--arrival-window", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "coding_swarm.csv"))
    a = ap.parse_args()

    events, sids = extract(a.sessions, a.min_rounds, a.max_rounds, a.model,
                           a.idle_cap, a.arrival_window, a.seed)
    n = schema.write_events_csv(a.out, events)

    # quick provenance summary
    import collections
    bytes_by = collections.Counter()
    ops_by = collections.Counter()
    for e in events:
        if e.op == "write":
            bytes_by[e.plane] += e.phys_bytes
            ops_by[e.plane] += 1
    span = max(e.ts for e in events) if events else 0
    print(f"wrote {n} events from {len(sids)} sessions -> {a.out}")
    print(f"  model={a.model} kv_bytes/token={schema.MODELS[a.model].kv_bytes_per_token:,.0f}")
    print(f"  scenario span={span:,.1f}s")
    print("  WRITE bytes by plane:", {k: f"{v/1e9:.2f}GB" for k, v in sorted(bytes_by.items())})
    print("  WRITE ops   by plane:", dict(sorted(ops_by.items())))


if __name__ == "__main__":
    main()
