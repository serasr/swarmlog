"""Log schema for multi-agent workflows with cross-references.

Design principle: each agent writes its OWN log. There is no trusted central
log. Integrity therefore cannot come from any single log being correct -- it
comes from the set of logs having to be mutually consistent.

Entry kinds
-----------
SEND       agent a sent message m to agent b
RECV       agent b received message m from agent a
TOOL_CALL  agent a invoked a tool and recorded args/result digests
CLAIM      agent a asserted something (e.g. "task done"), optionally citing
           the msg_ids / tool_call_ids that justify it

Every entry carries a per-agent `seq` and a Lamport clock so that causal
ordering violations are detectable across logs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any


def digest(obj: Any) -> str:
    """Stable short content hash. Used to detect silent content edits."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


SEND = "SEND"
RECV = "RECV"
TOOL_CALL = "TOOL_CALL"
CLAIM = "CLAIM"

ENTRY_KINDS = (SEND, RECV, TOOL_CALL, CLAIM)


@dataclass
class Entry:
    """One line in one agent's log."""

    entry_id: str
    agent: str
    seq: int
    lamport: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    # --- ground truth, never visible to the checker -----------------------
    tainted: bool = False  # True if this entry is part of the hidden attack
    forged: bool = False  # True if the tamperer fabricated this entry

    def copy(self) -> "Entry":
        return Entry(
            entry_id=self.entry_id,
            agent=self.agent,
            seq=self.seq,
            lamport=self.lamport,
            kind=self.kind,
            payload=json.loads(json.dumps(self.payload, default=str)),
            tainted=self.tainted,
            forged=self.forged,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SwarmLog:
    """The full set of per-agent logs, plus ground truth for scoring."""

    agents: list[str]
    logs: dict[str, list[Entry]]
    topology: str = "unknown"
    # ground truth bookkeeping
    attack_entry_ids: set[str] = field(default_factory=set)
    attack_agents: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    def all_entries(self) -> list[Entry]:
        out: list[Entry] = []
        for a in self.agents:
            out.extend(self.logs.get(a, []))
        return out

    def by_id(self) -> dict[str, Entry]:
        return {e.entry_id: e for e in self.all_entries()}

    def copy(self) -> "SwarmLog":
        return SwarmLog(
            agents=list(self.agents),
            logs={a: [e.copy() for e in self.logs[a]] for a in self.logs},
            topology=self.topology,
            attack_entry_ids=set(self.attack_entry_ids),
            attack_agents=set(self.attack_agents),
        )

    def stats(self) -> dict[str, int]:
        entries = self.all_entries()
        counts = {k: 0 for k in ENTRY_KINDS}
        for e in entries:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        counts["TOTAL"] = len(entries)
        counts["TAINTED"] = sum(1 for e in entries if e.tainted)
        return counts

    def to_jsonl(self, path: str, include_ground_truth: bool = True) -> None:
        with open(path, "w") as fh:
            for e in sorted(self.all_entries(), key=lambda x: (x.lamport, x.agent, x.seq)):
                d = e.to_dict()
                if not include_ground_truth:
                    d.pop("tainted", None)
                    d.pop("forged", None)
                fh.write(json.dumps(d) + "\n")
