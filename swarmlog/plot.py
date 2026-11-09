"""Figures for the report."""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "naive": "#4C78A8",
    "careful": "#E45756",
    "forging": "#72B7B2",
}
MARKER = {"informed": "o", "random": "s"}


def _load(csv_path: str) -> list[dict]:
    with open(csv_path) as fh:
        return list(csv.DictReader(fh))


def detection_curve(csv_path: str, out_png: str,
                    metric: str = "tamper_detected",
                    title: str | None = None,
                    strategies: tuple[str, ...] | None = None) -> str:
    rows = _load(csv_path)
    buckets: dict[tuple, list[float]] = defaultdict(list)
    closure: list[float] = []
    n_agents = 1
    for r in rows:
        if strategies and r["strategy"] not in strategies:
            continue
        n_agents = int(r["n_agents"])
        key = (r["strategy"], r["selection"], float(r["frac"]))
        buckets[key].append(1.0 if r[metric] == "True" else 0.0)
        closure.append(float(r["closure_size"]) / n_agents)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)

    series: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    for (strat, sel, frac), vals in buckets.items():
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        se = (var / len(vals)) ** 0.5  # standard error, not sd
        series[(strat, sel)].append((frac, mean, se))

    for (strat, sel), pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, label=f"{strat} / {sel}",
                    color=PALETTE.get(strat, "#777"),
                    marker=MARKER.get(sel, "^"), markersize=4.5,
                    linewidth=1.8, capsize=2.5,
                    linestyle="-" if sel == "informed" else "--", alpha=0.9)

    if closure:
        c = sum(closure) / len(closure)
        ax.axvline(c, color="#555", linestyle=":", linewidth=1.4)
        ax.text(c + 0.012, 0.52, "predicted\nclosure threshold",
                fontsize=8, color="#555", va="center")

    ax.set_xlabel("fraction of the swarm colluding")
    ax.set_ylabel({"tamper_detected": "P(tampering detected)",
                   "attack_implicated": "P(attack implicated)"}[metric])
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_title(title or "Cross-corroboration survives partial collusion",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def rule_breakdown(csv_path: str, out_png: str) -> str:
    """Which rules carry the signal, as collusion grows."""
    import json

    rows = _load(csv_path)
    per_frac: dict[float, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        if r["strategy"] != "careful" or r["selection"] != "informed":
            continue
        rules = json.loads(r["rules"])
        frac = float(r["frac"])
        for rule in ("DANGLING_RECV", "DANGLING_SEND", "DIGEST_MISMATCH",
                     "DANGLING_PROVENANCE", "DANGLING_CITATION",
                     "ORPHAN_BODY_REF", "UNSOURCED_TOOL_INPUT"):
            per_frac[frac][rule].append(rules.get(rule, 0))

    if not per_frac:
        return ""

    fracs = sorted(per_frac)
    rules = ["DANGLING_RECV", "DANGLING_SEND", "DIGEST_MISMATCH",
             "DANGLING_PROVENANCE", "DANGLING_CITATION",
             "ORPHAN_BODY_REF", "UNSOURCED_TOOL_INPUT"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756",
              "#B279A2", "#72B7B2", "#9D755D"]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    bottom = [0.0] * len(fracs)
    for rule, col in zip(rules, colors):
        vals = [sum(per_frac[f][rule]) / max(1, len(per_frac[f][rule]))
                for f in fracs]
        ax.bar([f"{f:.2f}" for f in fracs], vals, bottom=bottom,
               label=rule.lower().replace("_", " "), color=col, width=0.72)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xlabel("fraction of the swarm colluding")
    ax.set_ylabel("mean findings per run")
    ax.set_title("Which corroboration rule catches it (careful adversary)",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.2, axis="y", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
