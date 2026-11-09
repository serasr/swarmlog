#!/usr/bin/env python3
"""Competence curve, plus the live LLM run when a key is present.

    python3 run_cover_story.py                 # curve only (free)
    ANTHROPIC_API_KEY=... python3 run_cover_story.py --live --trials 10

The curve interpolates between tamper.py's `naive` (competence 0) and
`careful` (competence 1). A live run lands one point on it, and
estimate_competence() reads off the x-value.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from swarmlog.checker import check
from swarmlog.cover_story import (CoverStoryAdversary, competence_tamper,
                                  estimate_competence)
from swarmlog.experiment import score_run
from swarmlog.tamper import select_colluders
from swarmlog.workflow import WorkflowConfig, WorkflowGenerator

COMPETENCES = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]


def build(seed: int, n_agents: int, leak: float):
    cfg = WorkflowConfig(n_agents=n_agents, topology="random", rounds=12,
                         n_attack_agents=3, attack_rounds=4, p_leak=leak,
                         seed=seed)
    return WorkflowGenerator(cfg).run()


def competence_curve(n_agents: int, k: int, leak: float,
                     n_seeds: int) -> list[tuple[float, float]]:
    curve = []
    for comp in COMPETENCES:
        det = []
        for s in range(n_seeds):
            log = build(s, n_agents, leak)
            colluders = select_colluders(log, k, mode="informed", seed=s)
            res = competence_tamper(log, colluders, competence=comp, seed=s)
            rep = check(res.log)
            d, _ = score_run(rep, res.deleted_entity_ids)
            det.append(1.0 if d else 0.0)
        curve.append((comp, statistics.mean(det)))
        print(f"  competence={comp:.1f}  detection={curve[-1][1]:.2f}")
    return curve


def live_run(n_agents: int, k: int, leak: float, trials: int,
             model: str, outdir: str = "out") -> dict:
    """One request per trial. Results are appended to disk as they land, so a
    crash or a rate limit halfway through costs you nothing already paid for."""
    rates, violations, parse_fail, shrink = [], [], 0, []
    no_edit = 0
    trace_path = os.path.join(outdir, "live_trials.jsonl")

    for s in range(trials):
        log = build(s, n_agents, leak)
        colluders = select_colluders(log, k, mode="informed", seed=s)
        adv = CoverStoryAdversary(colluders, model=model)
        try:
            res = adv.rewrite(log)
        except Exception as exc:  # rate limit, overload, anything
            print(f"  seed {s}: CALL FAILED ({type(exc).__name__}: "
                  f"{str(exc)[:90]}) -- skipping")
            violations.append(f"seed {s} call failed: {type(exc).__name__}")
            continue

        rep = check(res.log)
        d, _ = score_run(rep, res.deleted_entity_ids)
        violations.extend(res.violations)
        parse_fail += 0 if res.parse_ok else 1

        # A trial only counts if the adversary actually changed the logs.
        # An unparsed or empty response leaves them untouched, which trivially
        # produces zero findings -- scoring that as evasion would invert the
        # result. Those trials are reported separately, never averaged in.
        usable = res.parse_ok and res.n_out < res.n_in
        if usable:
            rates.append(1.0 if d else 0.0)
            if res.n_in:
                shrink.append(res.n_out / res.n_in)
        else:
            no_edit += 1

        with open(trace_path, "a") as fh:
            fh.write(json.dumps({
                "seed": s, "detected": d, "parse_ok": res.parse_ok,
                "n_in": res.n_in, "n_out": res.n_out,
                "structural_findings": len(rep.structural()),
                "rules": rep.by_rule(), "violations": res.violations,
                "response_head": res.raw_response[:400],
            }) + "\n")

        flag = "" if usable else "   <-- NOT SCORED (no edits applied)"
        print(f"  seed {s}: detected={d} parse_ok={res.parse_ok} "
              f"kept={res.n_out}/{res.n_in} "
              f"findings={len(rep.structural())}{flag}")

    if not rates:
        return {"trials": 0, "model": model, "detection_rate": None,
                "parse_failures": parse_fail, "no_edit_trials": no_edit,
                "threat_model_violations": violations[:20],
                "note": ("no trial produced a usable cover story; the model "
                         "either refused, returned nothing parseable, or "
                         "proposed no edits. This is NOT an evasion result.")}
    return {
        "trials": len(rates),
        "model": model,
        "detection_rate": statistics.mean(rates),
        "parse_failures": parse_fail,
        "no_edit_trials": no_edit,
        "mean_log_retention": statistics.mean(shrink) if shrink else None,
        "threat_model_violations": violations[:20],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=10)
    p.add_argument("--k", type=int, default=6,
                   help="colluding set size; 6/10 spans the full curve")
    p.add_argument("--leak", type=float, default=0.5)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--live", action="store_true")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--model", default="claude-sonnet-4-5")
    p.add_argument("--outdir", default="out")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"competence curve (n={args.agents}, k={args.k}, "
          f"p_leak={args.leak}, {args.seeds} seeds)")
    curve = competence_curve(args.agents, args.k, args.leak, args.seeds)

    result = {"curve": curve, "k": args.k, "n_agents": args.agents,
              "p_leak": args.leak, "seeds": args.seeds}

    # ---- what the live run will cost ---------------------------------
    probe = build(0, args.agents, args.leak)
    probe_c = select_colluders(probe, args.k, mode="informed", seed=0)
    prompt_chars = len(CoverStoryAdversary._serialise(probe, probe_c))
    est_in = prompt_chars / 3.5  # rough chars-per-token
    print(f"\nlive run would send ~{est_in:,.0f} input tokens per trial "
          f"({args.trials} trials)")

    live = None
    if args.live:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("--live requested but ANTHROPIC_API_KEY is unset; "
                  "skipping the live run")
        else:
            print(f"\nlive run ({args.model}, {args.trials} trials)")
            live = live_run(args.agents, args.k, args.leak, args.trials,
                            args.model, outdir=args.outdir)
            est = (estimate_competence(live["detection_rate"], curve)
                   if live["detection_rate"] is not None else None)
            live["estimated_competence"] = est
            result["live"] = live
            if live["detection_rate"] is not None:
                print(f"\n  detection rate   {live['detection_rate']:.2f} "
                      f"({live['trials']} usable trials, "
                      f"{live.get('no_edit_trials', 0)} discarded)")
                print(f"  → effective competence "
                      f"{est if est is None else round(est, 2)}")
            else:
                print(f"\n  NO USABLE TRIALS ({live.get('no_edit_trials', 0)} "
                      f"produced no edits, {live['parse_failures']} failed to "
                      f"parse)")
                print("  This is not an evasion result. Inspect "
                      "out/live_trials.jsonl before interpreting anything.")
            if live.get("threat_model_violations"):
                print(f"  violations: {live['threat_model_violations'][:3]}")

    with open(os.path.join(args.outdir, "cover_story.json"), "w") as fh:
        json.dump(result, fh, indent=1)

    # ---- how much does ONE missed repair cost? -----------------------
    print("\ncost of an imperfect cover story")
    miss_curve, req = [], []
    for miss in range(0, 7):
        det = []
        for s in range(args.seeds):
            log = build(s, args.agents, args.leak)
            colluders = select_colluders(log, args.k, mode="informed", seed=s)
            res = competence_tamper(log, colluders, seed=s, miss_exactly=miss)
            d, _ = score_run(check(res.log), res.deleted_entity_ids)
            det.append(1.0 if d else 0.0)
            req.append(getattr(res, "n_repairs_required", 0))
        miss_curve.append((miss, statistics.mean(det)))
        print(f"  missed={miss}  detection={miss_curve[-1][1]:.2f}")
    result["miss_curve"] = miss_curve
    result["mean_repairs_required"] = statistics.mean(req)
    print(f"  (mean repairs required per cover story: "
          f"{statistics.mean(req):.1f})")

    # ---- figures -----------------------------------------------------
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=160)
    xs = [c for c, _ in curve]
    ys = [r for _, r in curve]
    ax.plot(xs, ys, marker="o", color="#4C78A8", linewidth=2,
            label="programmatic adversary")
    ax.annotate("naive", (0, ys[0]), textcoords="offset points",
                xytext=(8, -4), fontsize=8.5, color="#555")
    ax.annotate("careful", (1, ys[-1]), textcoords="offset points",
                xytext=(-38, 8), fontsize=8.5, color="#555")
    if live and live.get("estimated_competence") is not None:
        ax.scatter([live["estimated_competence"]], [live["detection_rate"]],
                   color="#E45756", s=70, zorder=5,
                   label=f"{live['model']} ({live['trials']} trials)")
        ax.axvline(live["estimated_competence"], color="#E45756",
                   linestyle=":", linewidth=1.2)
    ax.set_xlabel("adversary competence (P of performing each required repair)")
    ax.set_ylabel("P(tampering detected)")
    ax.set_ylim(-0.04, 1.06)
    ax.set_title(f"How good do colluders have to be? "
                 f"(k={args.k}/{args.agents})", fontsize=11, loc="left")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    mx = [m for m, _ in miss_curve]
    my = [r for _, r in miss_curve]
    axm.bar([str(m) for m in mx], my, color="#72B7B2", width=0.68)
    axm.set_xlabel(f"repairs missed (of ~{statistics.mean(req):.0f} required)")
    axm.set_ylabel("P(tampering detected)")
    axm.set_ylim(0, 1.0)
    axm.set_title("One slip is expensive", fontsize=11, loc="left")
    axm.grid(alpha=0.2, axis="y", linewidth=0.6)
    for s in ("top", "right"):
        axm.spines[s].set_visible(False)

    fig.tight_layout()
    out_png = os.path.join(args.outdir, "fig6_competence.png")
    fig.savefig(out_png)
    print(f"\nwrote {os.path.join(args.outdir, 'cover_story.json')}, {out_png}")


if __name__ == "__main__":
    main()