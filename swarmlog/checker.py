"""Cross-reference consistency checker.

The checker is BLIND: it never sees ground truth. It only sees the set of
per-agent logs and looks for places where they fail to corroborate each other.

Think double-entry bookkeeping. Every inter-agent event is written down twice,
by two parties with different incentives. A tamperer who edits one copy
produces a discrepancy; a tamperer who wants no discrepancy must control every
log that references the event, directly or transitively.

Rules
-----
R1 DANGLING_RECV        a receipt with no matching send
R2 DANGLING_SEND        a send with no matching receipt
R3 DIGEST_MISMATCH      both sides logged it, they disagree on the content
R4 CAUSAL_VIOLATION     receipt is not strictly after the send (Lamport)
R5 DANGLING_PROVENANCE  an artifact derived from something that doesn't exist
R6 DANGLING_CITATION    a claim citing an entry that doesn't exist
R7 SEQ_GAP              an agent's own sequence numbers are not contiguous
R8 LAMPORT_REGRESSION   an agent's clock runs backwards within its own log
R9 UNSOURCED_TOOL_INPUT a tool call derived from a message the agent never got
R10 ORPHAN_BODY_REF     a message body quotes an identifier that exists in no
                        log -- the rule that bites hardest, because honest
                        agents quote artifact ids the colluders deleted and
                        the colluders cannot edit honest logs

R7/R8 catch a sloppy tamperer. R1-R6 and R9 catch a careful one, and are the
rules that carry the signal in the headline experiment.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .schema import CLAIM, RECV, SEND, TOOL_CALL, Entry, SwarmLog

ID_PATTERN = re.compile(r"^[mte]\d{5}$")

SEVERITY = {
    "DANGLING_RECV": "high",
    "DANGLING_SEND": "high",
    "DIGEST_MISMATCH": "high",
    "CAUSAL_VIOLATION": "medium",
    "DANGLING_PROVENANCE": "high",
    "DANGLING_CITATION": "medium",
    "SEQ_GAP": "low",
    "LAMPORT_REGRESSION": "low",
    "UNSOURCED_TOOL_INPUT": "medium",
    "ORPHAN_BODY_REF": "high",
}

# Rules a careful adversary cannot trivially launder by renumbering.
STRUCTURAL_RULES = {
    "DANGLING_RECV", "DANGLING_SEND", "DIGEST_MISMATCH",
    "DANGLING_PROVENANCE", "DANGLING_CITATION", "UNSOURCED_TOOL_INPUT",
    "ORPHAN_BODY_REF",
}


def _scan_ids(obj) -> set[str]:
    """Pull anything that looks like an entity id out of a message body."""
    found: set[str] = set()
    if isinstance(obj, str):
        if ID_PATTERN.match(obj):
            found.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _scan_ids(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found |= _scan_ids(v)
    return found


@dataclass
class Finding:
    rule: str
    severity: str
    description: str
    entry_ids: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    # ids that were referenced but are missing from every log
    missing_refs: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"<{self.rule} {self.agents} {self.description}>"


@dataclass
class CheckReport:
    findings: list[Finding]

    @property
    def n(self) -> int:
        return len(self.findings)

    def by_rule(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for f in self.findings:
            out[f.rule] += 1
        return dict(out)

    def structural(self) -> list[Finding]:
        return [f for f in self.findings if f.rule in STRUCTURAL_RULES]

    def implicated_agents(self) -> set[str]:
        out: set[str] = set()
        for f in self.findings:
            out.update(f.agents)
        return out

    def missing_ref_ids(self) -> set[str]:
        out: set[str] = set()
        for f in self.findings:
            out.update(f.missing_refs)
        return out


def check(log: SwarmLog, rules: set[str] | None = None) -> CheckReport:
    entries = log.all_entries()
    findings: list[Finding] = []
    enabled = rules if rules is not None else set(SEVERITY)

    def emit(rule: str, desc: str, entry_ids=None, agents=None, missing=None):
        if rule not in enabled:
            return
        findings.append(Finding(
            rule=rule,
            severity=SEVERITY[rule],
            description=desc,
            entry_ids=list(entry_ids or []),
            agents=list(agents or []),
            missing_refs=list(missing or []),
        ))

    # ---- indexes -----------------------------------------------------
    sends: dict[str, Entry] = {}
    recvs: dict[str, Entry] = {}
    tool_calls: dict[str, Entry] = {}
    recv_by_agent: dict[str, set[str]] = defaultdict(set)

    for e in entries:
        if e.kind == SEND:
            sends[e.payload["msg_id"]] = e
        elif e.kind == RECV:
            recvs[e.payload["msg_id"]] = e
            recv_by_agent[e.agent].add(e.payload["msg_id"])
        elif e.kind == TOOL_CALL:
            tool_calls[e.payload["call_id"]] = e

    known_ids: set[str] = set(sends) | set(recvs) | set(tool_calls)
    known_ids |= {e.entry_id for e in entries}

    # ---- R1 / R2 / R3 / R4: the SEND/RECV ledger ---------------------
    for mid, r in recvs.items():
        s = sends.get(mid)
        if s is None:
            emit("DANGLING_RECV",
                 f"{r.agent} logged receipt of {mid} from "
                 f"{r.payload.get('from')}, but no sender logged it",
                 [r.entry_id], [r.agent, r.payload.get("from", "?")], [mid])
            continue
        if s.payload.get("content_digest") != r.payload.get("content_digest"):
            emit("DIGEST_MISMATCH",
                 f"{s.agent} and {r.agent} disagree on the content of {mid}",
                 [s.entry_id, r.entry_id], [s.agent, r.agent])
        if r.lamport <= s.lamport:
            emit("CAUSAL_VIOLATION",
                 f"{mid} received at t={r.lamport} but sent at t={s.lamport}",
                 [s.entry_id, r.entry_id], [s.agent, r.agent])

    for mid, s in sends.items():
        if mid not in recvs:
            emit("DANGLING_SEND",
                 f"{s.agent} logged sending {mid} to "
                 f"{s.payload.get('to')}, but the recipient has no record",
                 [s.entry_id], [s.agent, s.payload.get("to", "?")], [mid])

    # ---- R5: provenance ----------------------------------------------
    for e in entries:
        parent = e.payload.get("derived_from")
        if parent and parent not in known_ids:
            emit("DANGLING_PROVENANCE",
                 f"{e.agent}'s {e.kind.lower()} derives from {parent}, "
                 f"which exists in no log",
                 [e.entry_id], [e.agent], [parent])

    # ---- R6: citations -------------------------------------------------
    for e in entries:
        if e.kind != CLAIM:
            continue
        for cid in e.payload.get("cites", []):
            if cid not in known_ids:
                emit("DANGLING_CITATION",
                     f"{e.agent} cites {cid} in a claim, but it exists "
                     f"in no log",
                     [e.entry_id], [e.agent], [cid])

    # ---- R9: tool inputs the agent never received ----------------------
    for cid, t in tool_calls.items():
        parent = t.payload.get("derived_from")
        if not parent or not parent.startswith("m"):
            continue
        if parent in known_ids and parent not in recv_by_agent[t.agent]:
            # the message exists, but this agent has no receipt for it
            emit("UNSOURCED_TOOL_INPUT",
                 f"{t.agent} ran {t.payload.get('tool')} on input from "
                 f"{parent}, which it never logged receiving",
                 [t.entry_id], [t.agent], [])

    # ---- R10: identifiers quoted inside message bodies -----------------
    for e in entries:
        body = e.payload.get("body")
        if not body:
            continue
        for ref in sorted(_scan_ids(body)):
            if ref not in known_ids:
                emit("ORPHAN_BODY_REF",
                     f"{e.agent}'s log quotes {ref} in the body of "
                     f"{e.payload.get('msg_id', e.entry_id)}, but {ref} "
                     f"exists in no log",
                     [e.entry_id], [e.agent], [ref])

    # ---- R7 / R8: per-agent internal consistency -----------------------
    for a in log.agents:
        own = sorted(log.logs.get(a, []), key=lambda x: x.seq)
        for i, e in enumerate(own, start=1):
            if e.seq != i:
                emit("SEQ_GAP",
                     f"{a}'s log jumps to seq={e.seq} at position {i}",
                     [e.entry_id], [a])
                break
        last = -1
        for e in own:
            if e.lamport < last:
                emit("LAMPORT_REGRESSION",
                     f"{a}'s clock goes backwards at {e.entry_id}",
                     [e.entry_id], [a])
                break
            last = e.lamport

    return CheckReport(findings=findings)
