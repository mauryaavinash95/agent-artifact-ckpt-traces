"""
F9: C2 consolidation compaction. Cognitive memory is written by distilling raw
context into a much smaller consolidated form. We measure the ratio of raw bytes
to consolidated bytes on two real datasets: LoCoMo (a session's dialogue -> its
session summary) and PerLTQA (an event's content -> its summary). Consolidation
shrinks memory by roughly 3x to 30x, which is why periodic consolidation is a
lifecycle operation the substrate must support.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import schema

HERE = os.path.dirname(os.path.abspath(__file__))
LOCOMO = os.path.join(HERE, "data", "raw", "locomo", "locomo10.json")
PERLTQA = os.path.join(HERE, "data", "raw", "perltqa", "perltmem_en.json")


def locomo_compaction():
    d = json.load(open(LOCOMO))
    raw, cons = [], []
    for s in d:
        conv = s["conversation"]; summ = s.get("session_summary", {})
        sess = [k for k in conv if k.startswith("session_") and not k.endswith("date_time")]
        r = sum(len(str(t.get("text", ""))) for k in sess if isinstance(conv[k], list) for t in conv[k])
        c = sum(len(str(v)) for v in summ.values())
        if c > 0:
            raw.append(r); cons.append(c)
    return np.array(raw, float), np.array(cons, float)


def perltqa_compaction():
    d = json.load(open(PERLTQA))
    raw, cons = [], []
    for x in d:
        ev = x.get("events", {})
        for e in (ev.values() if isinstance(ev, dict) else ev):
            if isinstance(e, dict) and e.get("content") and e.get("summary"):
                raw.append(len(str(e["content"]))); cons.append(len(str(e["summary"])))
    return np.array(raw, float), np.array(cons, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "f9_c2_compaction"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cg = schema.plane_colors()["C2"]
    lr, lc = locomo_compaction()
    pr, pc = perltqa_compaction()
    sources = [
        ("LoCoMo\n(dialogue$\\to$summary)", lr.sum() / len(lr), lc.sum() / len(lc), lr.sum() / lc.sum()),
        ("PerLTQA\n(event$\\to$summary)", pr.mean(), pc.mean(), (pr / np.maximum(pc, 1)).mean()),
    ]

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(sources)); w = 0.38
    raws = [s[1] for s in sources]; conss = [s[2] for s in sources]
    ax.bar(x - w / 2, raws, w, color="0.55", edgecolor="black", lw=0.6, label="raw content")
    ax.bar(x + w / 2, conss, w, color=cg, edgecolor="black", lw=0.6, label="consolidated")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in sources], fontsize=8)
    ax.set_ylabel("bytes (log)", fontweight="bold")
    for i, s in enumerate(sources):
        ax.annotate(f"{s[3]:.1f}$\\times$", xy=(i, max(s[1], s[2])),
                    xytext=(i, max(s[1], s[2]) * 1.4), ha="center",
                    fontsize=9, fontweight="bold", color=cg)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight", dpi=300)

    print("=== F9 ===")
    for s in sources:
        print(f"  {s[0].splitlines()[0]}: raw={s[1]:,.0f}B cons={s[2]:,.0f}B ratio={s[3]:.1f}x")


if __name__ == "__main__":
    main()
