"""The push side of recon memory: what every run gets without asking.

Two things are pushed — the operating rules (memory/recon_rules.md) and the
standing day-0 competition facts — by seeding the run's message list with
push_context() before run_agent starts. The frozen core loop needs no change:
push is a property of how a run is wired, not of the loop.

Facts live in code and rules live in markdown on purpose. The facts change
only if the team retargets the competition, which is a code change anyway; the
rules are policy a human admin must be able to open and fix in seconds without
touching Python. The pull side is the tools' job: recall_dataset_notes and the
findings table are fetched mid-run, only when the request needs them.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = _REPO_ROOT / "memory" / "recon_rules.md"

# Day-0 facts (docs/PLAN.md §1), pinned once and attached to every recon run:
# the fact you always want present, so no run ever re-asks what it is
# predicting or optimizes the wrong metric.
_STANDING_FACTS = """\
Standing facts (Playground S6E7):
- target column: health_condition, 3 classes — at-risk 85.9%, unhealthy 8.4%, fit 5.8%
- metric: balanced accuracy (mean per-class recall), so the 5.8% class weighs as much as the 85.9% one
- competition data lives under workspace/data (train.csv, test.csv, sample_submission.csv)"""


def push_context():
    """The [SYSTEM] message that seeds every recon run: rules, then facts.

    A missing rules file raises on purpose: a run without its operating rules
    must fail loudly at setup, not proceed ungoverned. This is run wiring, not
    a tool body, so raising here can never leak a traceback into a transcript.
    """
    rules = RULES_PATH.read_text(encoding="utf-8").strip()
    return "[SYSTEM]\n%s\n\n%s" % (rules, _STANDING_FACTS)
