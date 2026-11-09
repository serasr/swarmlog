"""Validate the checker BEFORE spending a cent on real agents.

The critical test is the first one: a checker that fires on honest logs is
worthless, and every later number depends on it being silent here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmlog.checker import check
from swarmlog.schema import SEND, RECV
from swarmlog.tamper import required_closure, select_colluders, tamper
from swarmlog.workflow import WorkflowConfig, WorkflowGenerator


def build(seed=0, **kw):
    cfg = WorkflowConfig(seed=seed, **kw)
    return WorkflowGenerator(cfg).run()


def test_no_false_positives_on_honest_logs():
    for seed in range(25):
        log = build(seed=seed)
        rep = check(log)
        assert not rep.structural(), (
            f"seed {seed}: checker fired on an untampered log: "
            f"{rep.structural()[:3]}"
        )


def test_deleting_a_send_leaves_a_dangling_receipt():
    log = build(seed=1)
    victim = next(e for e in log.all_entries() if e.kind == SEND)
    log.logs[victim.agent] = [
        e for e in log.logs[victim.agent] if e.entry_id != victim.entry_id
    ]
    rules = check(log).by_rule()
    assert rules.get("DANGLING_RECV", 0) >= 1


def test_deleting_a_receipt_leaves_a_dangling_send():
    log = build(seed=2)
    victim = next(e for e in log.all_entries() if e.kind == RECV)
    log.logs[victim.agent] = [
        e for e in log.logs[victim.agent] if e.entry_id != victim.entry_id
    ]
    rules = check(log).by_rule()
    assert rules.get("DANGLING_SEND", 0) >= 1


def test_editing_content_on_one_side_is_caught():
    log = build(seed=3)
    victim = next(e for e in log.all_entries() if e.kind == SEND)
    victim.payload["content_digest"] = "0000deadbeef0000"
    rules = check(log).by_rule()
    assert rules.get("DIGEST_MISMATCH", 0) >= 1


def test_partial_collusion_is_detected():
    """One colluder scrubbing a multi-agent attack cannot stay consistent."""
    hits = 0
    trials = 15
    for seed in range(trials):
        log = build(seed=seed)
        colluders = select_colluders(log, 1, mode="informed", seed=seed)
        res = tamper(log, colluders, strategy="careful", seed=seed)
        if check(res.log).structural():
            hits += 1
    assert hits >= trials - 1, f"only {hits}/{trials} single-colluder scrubs caught"


def test_full_closure_collusion_can_hide():
    """Sanity: the detector is not omnipotent. With the whole closure
    colluding, the cover story should hold. If this fails, the checker is
    picking up an artefact rather than real corroboration."""
    clean = 0
    trials = 15
    for seed in range(trials):
        log = build(seed=seed)
        closure = required_closure(log)
        res = tamper(log, closure, strategy="careful", seed=seed)
        if not check(res.log).structural():
            clean += 1
    assert clean >= trials * 0.6, (
        f"closure-sized collusion still caught in {trials - clean}/{trials} "
        f"runs -- check the closure definition"
    )


def test_naive_strategy_is_noisier_than_careful():
    log = build(seed=7)
    k = max(2, len(required_closure(log)) // 2)
    c = select_colluders(log, k, mode="informed", seed=7)
    naive = len(check(tamper(log, c, "naive", seed=7).log).findings)
    careful = len(check(tamper(log, c, "careful", seed=7).log).findings)
    assert naive >= careful


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
