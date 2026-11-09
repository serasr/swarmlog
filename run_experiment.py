#!/usr/bin/env python3
"""Run the full experiment and write results + figures.

    python3 run_experiment.py                  # default sweep
    python3 run_experiment.py --quick          # fast smoke run
    python3 run_experiment.py --agents 12 --seeds 20 --topology star
"""

from __future__ import annotations

import argparse
import json
import os

from swarmlog.experiment import aggregate, false_positive_check, sweep
from swarmlog.plot import detection_curve, rule_breakdown
from swarmlog.workflow import WorkflowConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--topology", default="random",
                   choices=["chain", "star", "complete", "random"])
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--attack-agents", type=int, default=3)
    p.add_argument("--attack-rounds", type=int, default=4)
    p.add_argument("--leak", type=float, default=0.5)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--outdir", default="out")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.seeds, args.agents = 5, 6

    os.makedirs(args.outdir, exist_ok=True)
    base = WorkflowConfig(
        n_agents=args.agents,
        topology=args.topology,
        rounds=args.rounds,
        n_attack_agents=args.attack_agents,
        attack_rounds=args.attack_rounds,
        p_leak=args.leak,
    )

    print("== false-positive baseline (honest logs) ==")
    fp = false_positive_check(base, n_seeds=min(args.seeds, 20))
    print(json.dumps(fp, indent=1))
    if fp["max_structural_findings"] > 0:
        print("!! checker fires on honest logs -- fix before trusting anything")

    print("\n== sweep ==")
    csv_path = os.path.join(args.outdir, "results.csv")
    rows = sweep(base, n_seeds=args.seeds, out_csv=csv_path)
    agg = aggregate(rows)

    with open(os.path.join(args.outdir, "aggregate.json"), "w") as fh:
        json.dump({"false_positive_baseline": fp, "aggregate": agg},
                  fh, indent=1)

    print(f"{'strategy|selection|k':30s} {'det':>5s} {'imp':>5s} {'finds':>7s}")
    for k, v in agg.items():
        print(f"{k:30s} {v['detection_rate']:5.2f} "
              f"{v['implication_rate']:5.2f} "
              f"{v['mean_structural_findings']:7.1f}")

    f1 = detection_curve(
        csv_path, os.path.join(args.outdir, "fig1_detection.png"),
        strategies=("careful",),
        title="A careful adversary needs the whole reference closure to hide")
    f2 = detection_curve(
        csv_path, os.path.join(args.outdir, "fig2_implication.png"),
        metric="attack_implicated", strategies=("careful",),
        title="Detecting tampering is easier than implicating the attack")
    f3 = rule_breakdown(csv_path,
                        os.path.join(args.outdir, "fig3_rules.png"))
    f4 = detection_curve(
        csv_path, os.path.join(args.outdir, "fig4_strategies.png"),
        title="Sloppy scrubbing is caught at any collusion level")
    print("\nwrote:", csv_path, f1, f2, f3, f4, sep="\n  ")


if __name__ == "__main__":
    main()
