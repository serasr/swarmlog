#!/usr/bin/env python3
"""How much does the detection threshold depend on the leakage rate?

p_leak is the probability that an attack artifact gets referenced by an honest
agent's log. It directly inflates the reference closure, so if the headline
0.52 threshold is just an artefact of p_leak=0.5 we need to know that before
claiming anything general.

Reports, for each p_leak:
  - mean closure size (theory)
  - the collusion fraction at which detection first drops below 0.5 (measured)
  - whether the two agree
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from swarmlog.experiment import single_run
from swarmlog.workflow import WorkflowConfig

LEAKS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_AGENTS = 10
N_SEEDS = 20


def crossing_point(fracs: list[float], rates: list[float],
                   level: float = 0.5) -> float | None:
    """First fraction where detection falls below `level`, interpolated."""
    for i in range(1, len(fracs)):
        if rates[i - 1] >= level > rates[i]:
            span = rates[i - 1] - rates[i]
            if span == 0:
                return fracs[i]
            t = (rates[i - 1] - level) / span
            return fracs[i - 1] + t * (fracs[i] - fracs[i - 1])
    return None


def main() -> None:
    os.makedirs("out", exist_ok=True)
    rows = []
    summary = {}

    for leak in LEAKS:
        per_k: dict[int, list[bool]] = defaultdict(list)
        closures: list[int] = []
        for k in range(0, N_AGENTS + 1):
            for s in range(N_SEEDS):
                cfg = WorkflowConfig(n_agents=N_AGENTS, topology="random",
                                     rounds=12, n_attack_agents=3,
                                     attack_rounds=4, p_leak=leak, seed=s)
                r = single_run(cfg, k, "careful", "informed", seed=s)
                per_k[k].append(r.tamper_detected)
                closures.append(r.closure_size)
                rows.append({"p_leak": leak, "k": k, "frac": r.frac,
                             "seed": s, "detected": r.tamper_detected,
                             "implicated": r.attack_implicated,
                             "closure": r.closure_size,
                             "structural": r.n_structural})

        fracs = [k / N_AGENTS for k in sorted(per_k)]
        rates = [statistics.mean([1.0 if d else 0.0 for d in per_k[k]])
                 for k in sorted(per_k)]
        mean_closure = statistics.mean(closures)
        xp = crossing_point(fracs, rates)
        summary[str(leak)] = {
            "mean_closure": mean_closure,
            "closure_frac": mean_closure / N_AGENTS,
            "measured_threshold": xp,
            "gap": None if xp is None else round(xp - mean_closure / N_AGENTS, 3),
            "detection_at_10pct": rates[1],
            "rates": dict(zip([f"{f:.1f}" for f in fracs],
                              [round(r, 3) for r in rates])),
        }
        print(f"p_leak={leak:<5} closure={mean_closure:4.1f} "
              f"({mean_closure / N_AGENTS:.2f})  "
              f"measured_threshold={xp if xp is None else round(xp, 3)}")

    with open("out/sensitivity_leak.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open("out/sensitivity_leak.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    # ---- figure -----------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160,
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    cmap = plt.get_cmap("viridis")
    for i, leak in enumerate(LEAKS):
        d = summary[str(leak)]
        xs = [float(k) for k in d["rates"]]
        ys = list(d["rates"].values())
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8,
                color=cmap(i / max(1, len(LEAKS) - 1)),
                label=f"p_leak = {leak}")
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1.2)
    ax.set_xlabel("fraction of the swarm colluding")
    ax.set_ylabel("P(tampering detected)")
    ax.set_title("Leakage sets the detection threshold", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    th = [summary[str(l)]["closure_frac"] for l in LEAKS]
    ms = [summary[str(l)]["measured_threshold"] for l in LEAKS]
    ax2.plot(LEAKS, th, marker="s", linewidth=1.8, color="#4C78A8",
             label="predicted (closure)")
    ax2.plot(LEAKS, [m if m is not None else float("nan") for m in ms],
             marker="o", linewidth=1.8, color="#E45756",
             label="measured (50% crossing)")
    ax2.set_xlabel("p_leak")
    ax2.set_ylabel("collusion fraction needed to hide")
    ax2.set_title("Theory tracks measurement", fontsize=11, loc="left")
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.set_ylim(0, 1.05)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig("out/fig5_leak_sensitivity.png")
    print("\nwrote out/sensitivity_leak.csv, out/sensitivity_leak.json, "
          "out/fig5_leak_sensitivity.png")


if __name__ == "__main__":
    main()
