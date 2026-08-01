"""Prompt assembly from composable blocks.

    build(action) = [ task_spec       ]  shared
                    [ cv_protocol     ]  shared
                    [ output_contract ]  shared — ONE constant, also used by the validators
                    [ context_docs    ]  shared — EDA + discussion summaries, budgeted
                    [ tree_summary    ]  shared — what was tried, scores, repeated failures
                    [ action_block    ]  ── draft   : avoid what has been tried
                                          ── improve : parent code + score
                                          ── debug   : parent code + status-specific failure

The first five blocks are identical across the three actions. Writing them as
three templates instead would mean the output contract lives in three places,
and a fix to one would silently leave the other two wrong — a class of bug that
shows up as "the debug action produces worse code" and is nearly invisible in a
tree of twenty nodes.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass

from kagglecracker.config import PROJECT_ROOT, Action, Settings
from kagglecracker.db.store import Node, Protocol
from kagglecracker.engine.contract import render_output_contract

SANDBOX_DIR = PROJECT_ROOT / "sandbox"


def sandbox_package_versions() -> str:
    """The exact versions inside the image, read from the lock the image installs.

    Told rather than guessed: the sandbox has no network, so a model that
    assumes an older API writes code that cannot be fixed by installing
    anything. pandas 3.x in particular differs from the 2.x idioms most models
    have seen far more of.
    """
    pyproject = tomllib.loads((SANDBOX_DIR / "pyproject.toml").read_text())
    wanted = {
        dep.split(">")[0].split("=")[0].split("[")[0].strip().lower()
        for dep in pyproject["project"]["dependencies"]
    }
    lock = tomllib.loads((SANDBOX_DIR / "uv.lock").read_text())
    versions = {
        pkg["name"].lower(): pkg["version"]
        for pkg in lock.get("package", [])
        if pkg["name"].lower() in wanted
    }
    return "\n".join(f"    - {n} {v}" for n, v in sorted(versions.items()))


@dataclass(frozen=True, slots=True)
class PromptBlocks:
    task_spec: str
    cv_protocol: str
    output_contract: str
    context_docs: str
    tree_summary: str
    action_block: str

    def render(self) -> str:
        parts = [
            self.task_spec,
            self.cv_protocol,
            self.output_contract,
            self.context_docs,
            self.tree_summary,
            self.action_block,
        ]
        return "\n\n".join(p.strip() for p in parts if p and p.strip())


class PromptBuilder:
    def __init__(self, settings: Settings, protocol: Protocol) -> None:
        self.settings = settings
        self.protocol = protocol
        self._package_versions = sandbox_package_versions()

    # -- shared blocks ----------------------------------------------------

    def task_spec(self) -> str:
        return f"""\
You are writing a Python file that solves a Kaggle tabular competition.

Competition: {self.settings.competition}
Goal: maximise the cross-validated score under the protocol below, on held-out data.

Return ONLY the contents of `train.py`, in one ```python code block. No prose
before or after it. The file is executed exactly as written, unmodified."""

    def cv_protocol(self) -> str:
        return f"""\
## Cross-validation protocol

{self.protocol.describe()}

Why this protocol and not another: {self.protocol.rationale}

Report `cv_score` as the mean across folds and `fold_scores` as the per-fold
values, in fold order. Do not substitute a different scheme — scores from a
different protocol are not comparable to the rest of the search and are discarded."""

    def output_contract(self) -> str:
        return render_output_contract(
            protocol_description=self.protocol.describe(),
            package_versions=self._package_versions,
            timeout_s=self.settings.node_timeout_s,
            api_notes=self.settings.contract_api_notes,
        )

    def context_docs(self) -> str:
        """EDA findings and discussion summaries, each behind its own flag.

        Both are off by default. Experiment C enabled them together and measured
        a significant regression, but it cannot attribute that to either one —
        so neither is presumed innocent, and the flags exist so the follow-up
        experiment is a config change rather than a code change.

        Unbounded context is also the real cost curve as a tree grows: every
        node pays for every document, so a per-node estimate taken on day 2
        quietly understates the day-4 bill.
        """
        docs: list[tuple[str, str]] = []

        if self.settings.inject_eda:
            eda = self.settings.context_dir / "eda_findings.md"
            if eda.is_file():
                docs.append(("EDA findings", eda.read_text()))

        if self.settings.inject_discussions:
            discussions = sorted(
                (self.settings.context_dir / "discussions").glob("*.md"), reverse=True
            )
            for path in discussions:
                docs.append((f"Discussion summary ({path.stem})", path.read_text()))

        if not docs:
            return ""

        budget = self.settings.context_char_budget
        kept: list[str] = []
        used = 0
        for title, body in docs:  # EDA first, then newest discussions
            block = f"### {title}\n\n{body.strip()}"
            if used + len(block) > budget:
                remaining = budget - used
                if remaining > 500:
                    kept.append(block[:remaining] + "\n...(truncated to fit context budget)")
                break
            kept.append(block)
            used += len(block)

        return "## Context\n\n" + "\n\n".join(kept)

    def tree_summary(self, nodes: list[Node]) -> str:
        """What the search has already learned. This is the agent's only memory.

        Two things it must carry: what was tried and how it scored, so a draft
        does not repeat a model; and which failures keep recurring, so the tree
        stops rediscovering the same crash. Without the second, nodes 5, 9 and
        11 can die identically and node 17 learns nothing from any of them.
        """
        if not nodes:
            return "## Search so far\n\nNothing has been tried yet. This is the first attempt."

        scored = sorted(
            (n for n in nodes if n.scored),
            key=lambda n: n.cv_score,
            reverse=self.protocol.higher_is_better,
        )
        lines = ["## Search so far", ""]

        if scored:
            lines.append(f"Best {len(scored[:10])} of {len(scored)} scored attempts:")
            for n in scored[:10]:
                spread = ""
                if len(n.fold_scores) > 1:
                    lo, hi = min(n.fold_scores), max(n.fold_scores)
                    spread = f"  folds {lo:.4f}–{hi:.4f}"
                lines.append(f"  node {n.id} ({n.action}): {n.cv_score:.5f}{spread}")
        else:
            lines.append("No attempt has produced a valid score yet.")

        failures = [n for n in nodes if not n.scored]
        if failures:
            counts: dict[str, int] = {}
            for n in failures:
                counts[n.failure_signature] = counts.get(n.failure_signature, 0) + 1
            lines += ["", f"Failures so far ({len(failures)} of {len(nodes)} attempts):"]
            for sig, count in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
                suffix = f" (x{count})" if count > 1 else ""
                lines.append(f"  {sig}{suffix}")
            lines.append("")
            lines.append("Do not reintroduce a failure that already appears above.")

        return "\n".join(lines)

    # -- action blocks ----------------------------------------------------

    def draft_block(self, nodes: list[Node]) -> str:
        del nodes  # what was tried is already in tree_summary; no need to repeat it
        return """\
## Your task: DRAFT

Write a new solution from scratch, taking a different approach from anything in
the search summary above. Prefer a different model family or a materially
different feature treatment rather than a small variation on what exists.

Keep this first version simple enough to finish comfortably inside the time
budget. A solid, complete solution scores; an ambitious one that gets killed
partway through does not."""

    def improve_block(self, parent: Node) -> str:
        return f"""\
## Your task: IMPROVE

Below is the current best solution, scoring {parent.cv_score:.5f} (node {parent.id}).
Make one substantive improvement and return the complete file — not a diff, not
a fragment.

State your single hypothesis in a one-line comment at the top of the file
(`# hypothesis: ...`), then implement it. Changing several things at once makes
the result unattributable, and the tree cannot learn from it.

```python
{parent.code}
```"""

    def debug_block(self, parent: Node) -> str:
        """Failure text varies by status, and that difference is the whole point.

        A crash has a traceback to quote. A timeout does not — the process was
        killed, so there is nothing to point at, and the fix is to make the code
        faster rather than to correct a line. A contract violation ran fine and
        broke the output format. Handing all three the same prompt makes two of
        them unfixable.
        """
        guidance = {
            "error": (
                "The script raised an exception. The traceback is below. Fix the "
                "cause and return the complete corrected file."
            ),
            "timeout": (
                f"The script was KILLED at the {self.settings.node_timeout_s}s wall-clock "
                "limit, so there is no traceback — nothing crashed. It was simply too "
                "slow to finish. Make it fit the budget: fewer boosting rounds, a "
                "cheaper model, early stopping, or subsampling during search. Do not "
                "hunt for a bug; there may not be one."
            ),
            "contract_violation": (
                "The script ran to completion but broke the output contract. The exact "
                "mismatch is below. Fix the output format — the modelling itself may "
                "be fine and is worth keeping."
            ),
        }.get(parent.status, "The attempt failed. Details below.")

        return f"""\
## Your task: DEBUG

{guidance}

Failure ({parent.status}) from node {parent.id}:
```
{(parent.error_text or "(no detail captured)").strip()[-4000:]}
```

The code that produced it:
```python
{parent.code}
```"""

    # -- assembly ---------------------------------------------------------

    def build(
        self, action: Action, *, nodes: list[Node], parent: Node | None = None
    ) -> PromptBlocks:
        if action == "draft":
            action_block = self.draft_block(nodes)
        elif action == "improve":
            if parent is None:
                raise ValueError("improve requires a parent node")
            action_block = self.improve_block(parent)
        elif action == "debug":
            if parent is None:
                raise ValueError("debug requires a parent node")
            action_block = self.debug_block(parent)
        else:
            raise ValueError(f"unknown action {action!r}")

        return PromptBlocks(
            task_spec=self.task_spec(),
            cv_protocol=self.cv_protocol(),
            output_contract=self.output_contract(),
            context_docs=self.context_docs(),
            tree_summary=self.tree_summary(nodes),
            action_block=action_block,
        )

    def render(
        self, action: Action, *, nodes: list[Node], parent: Node | None = None
    ) -> str:
        return self.build(action, nodes=nodes, parent=parent).render()


__all__ = ["PromptBuilder", "PromptBlocks", "sandbox_package_versions"]
