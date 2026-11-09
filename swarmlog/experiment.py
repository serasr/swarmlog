"""The headline experiment.

Sweep the colluding fraction and measure whether the surviving logs still
contradict each other.

Three outcomes are scored per run:

  tamper_detected     the checker raised at least one structural finding
  attack_implicated   at least one finding points at a deleted tainted entity,
                      i.e. an investigator is led to the attack, not just told
                      "something is off"
  clean               no findings at all; the cover story holds

Baselines reported alongside:
  - false positives on untampered logs (must be zero)
  - the `required_closure` size, which is the theoretical threshold
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass

from .checker import STRUCTURAL_RULES, check
from .tamper import required_closure, select_colluders, tamper
from .workflow import WorkflowConfig, WorkflowGenerator


@dataclass
class RunResult:
    seed: int
    topology: str
    n_agents: int
    strategy: str
    selection: str
    k: int
    frac: float
    n_findings: int
    n_structural: int
    tamper_detected: bool
    attack_implicated: bool
    closure_size: int
    rules: str


def score_run(report, vanished: set[str]) -> tuple[bool, bool]:
    structural = report.structural()
    detected = len(structural) > 0
    implicated = any(
        set(f.missing_refs) & vanished for f in structural
    )
    return detected, implicated


def single_run(cfg: WorkflowConfig, k: int, strategy: str, selection: str,
               seed: int) -> RunResult:
    gen = WorkflowGenerator(cfg)
    log = gen.run()

    closure = required_closure(log)
    colluders = select_colluders(log, k, mode=selection, seed=seed)
    res = tamper(log, colluders, strategy=strategy, seed=seed)
    report = check(res.log)

    detected, implicated = score_run(report, res.deleted_entity_ids)
    return RunResult(
        seed=cfg.seed,
        topology=cfg.topology,
        n_agents=cfg.n_agents,
        strategy=strategy,
        selection=selection,
        k=k,
        frac=k / cfg.n_agents,
        n_findings=report.n,
        n_structural=len(report.structural()),
        tamper_detected=detected,
        attack_implicated=implicated,
        closure_size=len(closure),
        rules=json.dumps(report.by_rule()),
    )


def false_positive_check(cfg: WorkflowConfig, n_seeds: int = 20) -> dict:
    """The checker must be silent on honest logs. Non-negotiable baseline."""
    totals = []
    rule_hits: dict[str, int] = {}
    for s in range(n_seeds):
        c = WorkflowConfig(**{**cfg.__dict__, "seed": s})
        log = WorkflowGenerator(c).run()
        rep = check(log)
        structural = rep.structural()
        totals.append(len(structural))
        for f in structural:
            rule_hits[f.rule] = rule_hits.get(f.rule, 0) + 1
    return {
        "seeds": n_seeds,
        "mean_structural_findings": statistics.mean(totals),
        "max_structural_findings": max(totals),
        "rule_hits": rule_hits,
    }


def sweep(base: WorkflowConfig, strategies=("naive", "careful", "forging"),
          selections=("informed", "random"), n_seeds: int = 10,
          out_csv: str = "results.csv") -> list[RunResult]:
    rows: list[RunResult] = []
    for strategy in strategies:
        for selection in selections:
            for k in range(0, base.n_agents + 1):
                for s in range(n_seeds):
                    cfg = WorkflowConfig(**{**base.__dict__, "seed": s})
                    rows.append(single_run(cfg, k, strategy, selection, seed=s))

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return rows


def aggregate(rows: list[RunResult]) -> dict:
    """Detection rate by (strategy, selection, k), with variance."""
    buckets: dict[tuple, list[RunResult]] = {}
    for r in rows:
        buckets.setdefault((r.strategy, r.selection, r.k), []).append(r)

    out = {}
    for key, rs in sorted(buckets.items()):
        det = [1.0 if r.tamper_detected else 0.0 for r in rs]
        imp = [1.0 if r.attack_implicated else 0.0 for r in rs]
        out["|".join(map(str, key))] = {
            "n": len(rs),
            "detection_rate": statistics.mean(det),
            "detection_sd": statistics.pstdev(det),
            "implication_rate": statistics.mean(imp),
            "mean_structural_findings": statistics.mean(
                [r.n_structural for r in rs]),
            "mean_closure_size": statistics.mean([r.closure_size for r in rs]),
        }
    return out
