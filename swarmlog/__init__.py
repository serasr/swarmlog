"""swarmlog -- cross-corroboration for tamper-resistant multi-agent oversight.

Built for the Apart Research / AI Alignment Foundation Swarm Alignment
Hackathon, November 2026, Track 4 (Multi-Agent Oversight and Control).

Research question: agents in the Hugging Face incident spoofed or deleted
their own transcripts in roughly 7% of reviewed cases (METR, 2026). Hash
chains stop retroactive editing but say nothing about lying at write time.
In a swarm, however, logs are redundant: every inter-agent event is recorded
twice by parties the adversary may not both control.

So: what fraction of a swarm must collude before it can produce an internally
consistent set of logs that still hides an attack?
"""

from .schema import Entry, SwarmLog, digest
from .workflow import WorkflowConfig, WorkflowGenerator, generate
from .checker import CheckReport, Finding, check
from .tamper import tamper, select_colluders, required_closure

__all__ = [
    "Entry", "SwarmLog", "digest",
    "WorkflowConfig", "WorkflowGenerator", "generate",
    "CheckReport", "Finding", "check",
    "tamper", "select_colluders", "required_closure",
]
