"""Which action to take next, and on which node.

Frozen by design. The design doc names policy tuning as this project's top risk
— it is an endless, seductive rabbit hole — so v1 is deliberately dumb and
stays that way for week 1.

    step < 3                      -> draft        (seed the tree with diversity)
    otherwise, sample:
        improve  p=0.70           -> best scored node
        draft    p=0.15
        debug    p=0.15           -> most recent repairable failure

    ┌── fallback (a): sampled action has no eligible node ──────────────┐
    │   improve with nothing scored yet  -> draft                       │
    │   debug with no repairable failure -> draft                       │
    ├── fallback (b): depth cap ────────────────────────────────────────┤
    │   improve targets the best node with depth < cap                  │
    │   if every scored node is at the cap -> draft                     │
    └───────────────────────────────────────────────────────────────────┘

Both fallbacks land on `draft`, which is always legal: it needs no parent. That
is what makes the policy total — there is no state in which it has nothing to
do, so the loop can never stall.

Seeded RNG throughout, so a run replays identically. That matters more than it
looks: when the tree is the thing under study, a search you cannot reproduce is
a search you cannot debug.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from typing import Literal

from kagglecracker.db.store import Node, Protocol
from kagglecracker.engine.ranking import best_node, most_recent_failure

Action = Literal["draft", "improve", "debug"]

#: Steps at the start that are always drafts, to seed the tree with different
#: model families rather than deepening one lucky first attempt.
SEED_DRAFTS = 3


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    parent: Node | None
    reason: str

    def __str__(self) -> str:
        target = f" node {self.parent.id}" if self.parent else ""
        return f"{self.action}{target} ({self.reason})"


@dataclass
class SearchPolicy:
    protocol: Protocol
    seed: int = 42
    max_depth: int = 5
    p_improve: float = 0.70
    p_draft: float = 0.15
    p_debug: float = 0.15
    seed_drafts: int = SEED_DRAFTS

    def __post_init__(self) -> None:
        total = self.p_improve + self.p_draft + self.p_debug
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"action probabilities must sum to 1.0, got {total}")
        self._rng = random.Random(self.seed)

    def reseed(self, step: int) -> None:
        """Make the RNG a function of (seed, step) rather than of call history.

        Without this, resuming a run at step 12 would draw from a fresh
        sequence and diverge from the same run executed uninterrupted. Deriving
        the stream from the step number makes resume produce the same decisions
        as if nothing had happened.
        """
        # A string seed, not a tuple: Random() rejects tuples, and str seeding
        # is derived from the bytes rather than from PYTHONHASHSEED, so it
        # reproduces across processes.
        self._rng = random.Random(f"{self.seed}:{step}")

    def decide(self, conn: sqlite3.Connection, *, step: int) -> Decision:
        self.reseed(step)

        if step < self.seed_drafts:
            return Decision("draft", None, f"seeding: step {step} of {self.seed_drafts}")

        roll = self._rng.random()
        if roll < self.p_improve:
            sampled: Action = "improve"
        elif roll < self.p_improve + self.p_draft:
            sampled = "draft"
        else:
            sampled = "debug"

        if sampled == "draft":
            return Decision("draft", None, "sampled draft")

        if sampled == "improve":
            # Fallback (b) is applied first: ask only for nodes we are allowed
            # to extend. Selecting the global best and then discovering it is at
            # the cap would leave the policy with nothing to do.
            parent = best_node(conn, self.protocol, max_depth=self.max_depth)
            if parent is None:
                # Distinguish "nothing scored yet" from "everything is at the
                # cap" — same fallback, very different things to see in a log.
                any_scored = best_node(conn, self.protocol) is not None
                reason = (
                    "sampled improve, but every scored node is at the depth cap"
                    if any_scored
                    else "sampled improve, but nothing has scored yet"
                )
                return Decision("draft", None, reason)
            return Decision("improve", parent, f"improving best (depth {parent.depth})")

        parent = most_recent_failure(conn, self.protocol, max_depth=self.max_depth)
        if parent is None:
            return Decision("draft", None, "sampled debug, but there is no repairable failure")
        return Decision("debug", parent, f"debugging {parent.status}")
